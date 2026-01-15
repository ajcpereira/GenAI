import asyncio
import signal
from pathlib import Path

import uvicorn
import yaml

from src.genai_core.runtime.state import RuntimeState
from src.genai_core.vllm.launcher import VLLMLauncher
from src.genai_core.api import create_app


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


async def amain(config_path: str = "config/config.yaml") -> int:
    cfg = load_config(config_path)

    runtime = RuntimeState()
    launcher = VLLMLauncher(cfg["vllm"], runtime=runtime)

    # Start vLLM and validate health
    await launcher.start_and_wait_healthy()

    # Derive and store runtime model limits (context window etc.)
    await launcher.populate_runtime_model_limits()

    # Build API (demo)
    app = create_app(cfg=cfg, runtime=runtime)

    # Run API server (core), while vLLM runs as a subprocess
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    )

    # Ensure clean shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_sig(*_):
        stop_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _handle_sig)
        except NotImplementedError:
            pass

    api_task = asyncio.create_task(server.serve())
    await stop_event.wait()

    await launcher.stop()
    server.should_exit = True
    await api_task
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
