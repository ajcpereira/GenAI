import asyncio
import os
import subprocess
from dataclasses import dataclass
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
    extra_args: Optional[List[str]] = None


class VLLMLauncher:
    def __init__(self, cfg: Dict[str, Any], runtime: RuntimeState):
        self.cfg = VLLMConfig(**cfg)
        self.runtime = runtime
        self.process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    def _build_cmd(self) -> List[str]:
        cmd = [
            "python",
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

        cmd = self._build_cmd()
        env = os.environ.copy()

        # Start subprocess
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # Wait for health endpoint
        await self._wait_healthy()

    @retry(stop=stop_after_delay(90), wait=wait_fixed(1))
    async def _wait_healthy(self) -> None:
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
