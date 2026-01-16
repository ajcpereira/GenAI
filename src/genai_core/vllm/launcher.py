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

from ..runtime.state import RuntimeState, ModelInfo
from .model_limits import derive_model_limits


log = logging.getLogger("genai_core.vllm_launcher")


@dataclass
class VLLMConfig:
    model_path: str
    served_model_name: str = "local-model"
    host: str = "127.0.0.1"
    port: int = 8001
    python_bin: Optional[str] = None
    log_file: str = "./logs/vllm.log"
    startup_timeout_s: int = 180
    extra_args: Optional[List[str]] = None


class VLLMLauncher:
    """
    Starts vLLM as a subprocess OR attaches to an already-running vLLM on the configured host/port.

    IMPORTANT: subprocess stdout reading must NOT block the asyncio event loop.
    We therefore read stdout in a background thread via asyncio.to_thread.
    """

    def __init__(self, cfg: Dict[str, Any], runtime: RuntimeState):
        self.cfg = VLLMConfig(**cfg)
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
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.cfg.model_path,
            "--served-model-name",
            self.cfg.served_model_name,
            "--host",
            self.cfg.host,
            "--port",
            str(self.cfg.port),
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
            raise RuntimeError("vLLM subprocess already started by this launcher instance")

        # Attach mode
        if await self._is_healthy():
            log.info("vLLM already healthy at %s - attaching (no subprocess spawn).", self.base_url)
            self.runtime.vllm_health = {"status": "healthy", "url": self.base_url}
            self._owns_process = False
            return

        # Start subprocess
        Path(self.cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self.cfg.log_file, "a", encoding="utf-8")

        cmd = self._build_cmd()
        env = os.environ.copy()

        log.info("Starting vLLM subprocess: %s", " ".join(cmd))
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

        # Read stdout in a background thread to avoid blocking the event loop.
        self._log_task = asyncio.create_task(self._consume_stdout_threaded(), name="vllm_stdout_consumer")

        # Wait for health with a hard timeout
        try:
            await asyncio.wait_for(self._wait_healthy_loop(), timeout=self.cfg.startup_timeout_s)
        except asyncio.TimeoutError as e:
            tail = "\n".join(self._stdout_tail)
            raise RuntimeError(
                f"Timed out waiting for vLLM to become healthy after {self.cfg.startup_timeout_s}s.\n"
                f"Health URL: {self.base_url}/health\n"
                f"Last log lines:\n{tail}\n\nFull log file: {self.cfg.log_file}"
            ) from e
        except Exception as e:
            if self.process and self.process.poll() is not None:
                code = self.process.returncode
                tail = "\n".join(self._stdout_tail)
                raise RuntimeError(
                    f"vLLM process exited early (code={code}).\n"
                    f"Health URL: {self.base_url}/health\n"
                    f"Last log lines:\n{tail}\n\nFull log file: {self.cfg.log_file}"
                ) from e
            raise

    async def _wait_healthy_loop(self) -> None:
        while True:
            if self.process and self.process.poll() is not None:
                raise RuntimeError("vLLM terminated before becoming healthy")

            if await self._is_healthy():
                self.runtime.vllm_health = {"status": "healthy", "url": self.base_url}
                log.info("vLLM is healthy at %s", self.base_url)
                return

            log.info("Waiting for vLLM health at %s ...", self.base_url)
            await asyncio.sleep(1)

    async def _consume_stdout_threaded(self) -> None:
        """
        Run a blocking stdout reader in a thread and forward lines back here.
        """
        if not self.process or not self.process.stdout:
            return

        def _reader():
            while True:
                if self._stop_log_reader:
                    return
                line = self.process.stdout.readline()
                if not line:
                    return
                yield line

        try:
            # iterate lines in a thread; each iteration yields a line
            while True:
                line = await asyncio.to_thread(lambda: next(_reader(), None))
                if line is None:
                    break

                line = line.rstrip("\n")
                self._stdout_tail.append(line)
                if self._log_fp:
                    self._log_fp.write(line + "\n")
                    self._log_fp.flush()
        except Exception:
            # Best-effort: do not crash the core due to logging issues
            return

    async def populate_runtime_model_limits(self) -> None:
        limits = derive_model_limits(self.cfg.model_path)
        self.runtime.model_info = ModelInfo(
            model_name=self.cfg.served_model_name,
            model_limits=limits,
            tokenizer_name_or_path=self.cfg.model_path,
        )

    async def stop(self) -> None:
        try:
            self._stop_log_reader = True

            if self._owns_process and self.process is not None:
                log.info("Stopping vLLM subprocess (pid=%s)...", self.process.pid)
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
