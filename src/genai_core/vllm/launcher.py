import asyncio
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_delay, wait_fixed

from ..runtime.state import RuntimeState, ModelInfo
from .model_limits import derive_model_limits


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
    def __init__(self, cfg: Dict[str, Any], runtime: RuntimeState):
        self.cfg = VLLMConfig(**cfg)
        self.runtime = runtime
        self.process: Optional[subprocess.Popen] = None
        self._stdout_tail: deque[str] = deque(maxlen=200)
        self._log_task: Optional[asyncio.Task] = None
        self._log_fp = None

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

    async def start_and_wait_healthy(self) -> None:
        if self.process is not None:
            raise RuntimeError("vLLM is already running")

        Path(self.cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self.cfg.log_file, "a", encoding="utf-8")

        cmd = self._build_cmd()
        env = os.environ.copy()

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        self._log_task = asyncio.create_task(self._consume_stdout())

        try:
            await self._wait_healthy()
        except Exception as e:
            if self.process and self.process.poll() is not None:
                code = self.process.returncode
                tail = "\n".join(self._stdout_tail)
                raise RuntimeError(
                    f"vLLM process exited early (code={code}).\n"
                    f"Last log lines:\n{tail}\n\nFull log file: {self.cfg.log_file}"
                ) from e
            raise

    async def _consume_stdout(self) -> None:
        if not self.process or not self.process.stdout:
            return
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            self._stdout_tail.append(line)
            if self._log_fp:
                self._log_fp.write(line + "\n")
                self._log_fp.flush()

    @retry(stop=stop_after_delay(180), wait=wait_fixed(1))
    async def _wait_healthy(self) -> None:
        if self.process and self.process.poll() is not None:
            raise RuntimeError("vLLM terminated before becoming healthy")

        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{self.base_url}/health")
            r.raise_for_status()
            self.runtime.vllm_health = {"status": "healthy", "url": self.base_url}

    async def populate_runtime_model_limits(self) -> None:
        limits = derive_model_limits(self.cfg.model_path)
        self.runtime.model_info = ModelInfo(
            model_name=self.cfg.served_model_name,
            model_limits=limits,
            tokenizer_name_or_path=self.cfg.model_path,
        )

    async def stop(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
            await asyncio.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
        finally:
            self.process = None
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
