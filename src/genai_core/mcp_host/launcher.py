import asyncio
import logging
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from ..runtime.state import RuntimeState

log = logging.getLogger("genai_core.mcp_launcher")


@dataclass
class MCPHostConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    python_bin: Optional[str] = None
    log_file: str = "./logs/mcp_host.log"
    startup_timeout_s: int = 30
    extra_args: Optional[List[str]] = None
    # Import path for uvicorn to run (factory mode)
    app_factory: str = "src.genai_core.mcp_host.app:create_app"


class MCPHostLauncher:
    """
    Starts MCP Host (FastAPI) as a subprocess OR attaches to an already-running MCP Host on host/port.

    We intentionally keep MCP Host out-of-process for separation of concerns and blast-radius control.
    """

    def __init__(self, cfg: Dict[str, Any], runtime: RuntimeState):
        self.cfg = MCPHostConfig(**cfg)
        self.runtime = runtime

        self.process: Optional[subprocess.Popen] = None
        self._stdout_tail: deque[str] = deque(maxlen=200)
        self._log_task: Optional[asyncio.Task] = None
        self._log_fp = None

        self._owns_process: bool = False
        self._stop_log_reader: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    def _build_cmd(self) -> List[str]:
        python_bin = self.cfg.python_bin or sys.executable
        cmd = [
            python_bin,
            "-m",
            "uvicorn",
            self.cfg.app_factory,
            "--factory",
            "--host",
            self.cfg.host,
            "--port",
            str(self.cfg.port),
            "--log-level",
            "info",
        ]
        if self.cfg.extra_args:
            cmd.extend(self.cfg.extra_args)
        return cmd

    async def _is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def start_and_wait_healthy(self) -> None:
        if self.process is not None:
            raise RuntimeError("MCP Host subprocess already started by this launcher instance")

        # Attach mode
        if await self._is_healthy():
            log.info("MCP Host already healthy at %s - attaching (no subprocess spawn).", self.base_url)
            self.runtime.mcp_health = {"status": "healthy", "url": self.base_url}
            self._owns_process = False
            return

        # Start subprocess
        Path(self.cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self.cfg.log_file, "a", encoding="utf-8")

        cmd = self._build_cmd()
        env = os.environ.copy()

        log.info("Starting MCP Host subprocess: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._owns_process = True
        self._stop_log_reader = False

        self._log_task = asyncio.create_task(self._consume_stdout_threaded(), name="mcp_host_stdout_consumer")

        try:
            await asyncio.wait_for(self._wait_healthy_loop(), timeout=self.cfg.startup_timeout_s)
        except asyncio.TimeoutError as e:
            tail = "\n".join(self._stdout_tail)
            raise RuntimeError(
                f"Timed out waiting for MCP Host to become healthy after {self.cfg.startup_timeout_s}s.\n"
                f"Health URL: {self.base_url}/health\n"
                f"Last log lines:\n{tail}\n\nFull log file: {self.cfg.log_file}"
            ) from e
        except Exception as e:
            if self.process and self.process.poll() is not None:
                code = self.process.returncode
                tail = "\n".join(self._stdout_tail)
                raise RuntimeError(
                    f"MCP Host process exited early (code={code}).\n"
                    f"Health URL: {self.base_url}/health\n"
                    f"Last log lines:\n{tail}\n\nFull log file: {self.cfg.log_file}"
                ) from e
            raise

    async def _wait_healthy_loop(self) -> None:
        while True:
            if self.process and self.process.poll() is not None:
                raise RuntimeError("MCP Host terminated before becoming healthy")

            if await self._is_healthy():
                self.runtime.mcp_health = {"status": "healthy", "url": self.base_url}
                log.info("MCP Host is healthy at %s", self.base_url)
                return

            log.info("Waiting for MCP Host health at %s ...", self.base_url)
            await asyncio.sleep(0.5)

    async def _consume_stdout_threaded(self) -> None:
        if not self.process or not self.process.stdout:
            return

        def _readline():
            return self.process.stdout.readline()

        try:
            while True:
                if self._stop_log_reader:
                    return

                line = await asyncio.to_thread(_readline)
                if not line:
                    return

                line = line.rstrip("\n")
                self._stdout_tail.append(line)
                if self._log_fp:
                    self._log_fp.write(line + "\n")
                    self._log_fp.flush()
        except Exception:
            return

    async def stop(self) -> None:
        try:
            self._stop_log_reader = True

            if self._owns_process and self.process is not None:
                log.info("Stopping MCP Host subprocess (pid=%s)...", self.process.pid)
                self.process.terminate()
                await asyncio.sleep(0.5)
                if self.process.poll() is None:
                    self.process.kill()
        finally:
            self.process = None
            self._owns_process = False

            t = self._log_task
            if t:
                t.cancel()
                self._log_task = None

            if self._log_fp:
                try:
                    self._log_fp.close()
                except Exception:
                    pass
                self._log_fp = None
