import asyncio
import logging
import signal
from pathlib import Path

import uvicorn
import yaml

from src.genai_core.runtime.state import RuntimeState
from src.genai_core.vllm.launcher import VLLMLauncher
from src.genai_core.api import create_app
from src.genai_core.logging_setup import setup_logging


log = logging.getLogger("genai_core.main")


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


async def amain(config_path: str = "config/config.yaml") -> int:
    cfg = load_config(config_path)

    # Logging first, so any startup crash is visible.
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("core_log_file", "./logs/core.log"),
    )

    api_host = cfg.get("api", {}).get("host", "127.0.0.1")
    api_port = int(cfg.get("api", {}).get("port", 8000))

    runtime = RuntimeState()
    runtime.ready = False
    runtime.ready_reason = "starting"

    launcher = VLLMLauncher(cfg["vllm"], runtime=runtime)

    app = create_app(cfg=cfg, runtime=runtime)

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=api_host,
            port=api_port,
            log_level="info",
        )
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_sig(signame: str):
        log.info("Received %s - shutting down...", signame)
        stop_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, lambda ss=s: _handle_sig(ss.name))
        except NotImplementedError:
            pass

    # Start API server FIRST
    log.info("Starting Core API (FastAPI-first) on %s:%s", api_host, api_port)
    api_task = asyncio.create_task(server.serve(), name="core_api_server")

    def _api_done(task: asyncio.Task):
        try:
            task.result()
            log.error("Core API server stopped unexpectedly.")
        except Exception as e:
            log.exception("Core API server crashed: %s", str(e))
        finally:
            stop_event.set()

    api_task.add_done_callback(_api_done)

    # Warmup vLLM in background
    async def warmup_vllm():
        try:
            runtime.ready = False
            runtime.ready_reason = "starting_vllm"
            log.info("Warmup: starting/attaching vLLM in background...")

            await launcher.start_and_wait_healthy()
            runtime.ready_reason = "loading_model_limits"
            await launcher.populate_runtime_model_limits()

            runtime.ready = True
            runtime.ready_reason = "ready"
            log.info("Warmup complete: READY (vLLM=%s model=%s)", runtime.vllm_health, runtime.model_info.model_name if runtime.model_info else None)

        except Exception as e:
            runtime.ready = False
            runtime.ready_reason = f"warmup_failed: {e}"
            log.exception("Warmup failed: %s", str(e))

    warmup_task = asyncio.create_task(warmup_vllm(), name="vllm_warmup")

    # Wait for either a signal or API crash
    await stop_event.wait()

    # Cleanup
    log.info("Stopping vLLM (if owned)...")
    await launcher.stop()

    server.should_exit = True
    try:
        await api_task
    except Exception:
        pass

    # Stop warmup task if still running
    if not warmup_task.done():
        warmup_task.cancel()

    log.info("Shutdown complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
