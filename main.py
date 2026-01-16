import asyncio
import logging
import signal

import uvicorn

from src.genai_core.api import create_app
from src.genai_core.config.load import load_and_validate_config
from src.genai_core.logging_setup import setup_logging
from src.genai_core.mcp_host.launcher import MCPHostLauncher
from src.genai_core.runtime.state import RuntimeState
from src.genai_core.vllm.launcher import VLLMLauncher

log = logging.getLogger("genai_core.main")


async def amain(config_path: str = "config/config.yaml") -> int:
    cfg, cfg_model = load_and_validate_config(config_path)

    # Logging first, so any startup crash is visible.
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("core_log_file", "./logs/core.log"),
        rotation=log_cfg.get("rotation", {}),
    )

    api_host = cfg_model.api.host
    api_port = int(cfg_model.api.port)

    runtime = RuntimeState()
    runtime.ready = False
    runtime.ready_reason = "starting"

    vllm_launcher = VLLMLauncher(cfg["vllm"], runtime=runtime)

    # MCP Host launcher config (optional)
    tools_cfg = cfg.get("tools", {})
    mcp_cfg = tools_cfg.get("mcp", {})
    mcp_enabled = bool(mcp_cfg.get("enabled", False))

    # Default MCP host bind/port should match tools.mcp.base_url, but we keep it explicit for launcher.
    # If you want, you can derive host/port from base_url; for now keep config simple.
    mcp_launcher_cfg = cfg.get("mcp_host", {}) or {}
    mcp_launcher = MCPHostLauncher(cfg=mcp_launcher_cfg, runtime=runtime) if mcp_enabled else None

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

    # Warmup in background (MCP then vLLM)
    async def warmup():
        try:
            runtime.ready = False
            runtime.ready_reason = "starting_dependencies"

            # 1) MCP Host (best-effort; do not block readiness if it fails)
            if mcp_launcher is not None:
                try:
                    runtime.ready_reason = "starting_mcp_host"
                    log.info("Warmup: starting/attaching MCP Host in background...")
                    await mcp_launcher.start_and_wait_healthy()
                except Exception as e:
                    runtime.mcp_health = {"status": "unhealthy", "error": str(e)}
                    log.warning("MCP Host warmup failed (continuing without web): %s", str(e))

            # 2) vLLM (required for readiness)
            runtime.ready_reason = "starting_vllm"
            log.info("Warmup: starting/attaching vLLM in background...")
            await vllm_launcher.start_and_wait_healthy()

            runtime.ready_reason = "loading_model_limits"
            await vllm_launcher.populate_runtime_model_limits()

            runtime.ready = True
            runtime.ready_reason = "ready"
            log.info(
                "Warmup complete: READY (vLLM=%s model=%s MCP=%s)",
                runtime.vllm_health,
                runtime.model_info.model_name if runtime.model_info else None,
                runtime.mcp_health,
            )

        except Exception as e:
            runtime.ready = False
            runtime.ready_reason = f"warmup_failed: {e}"
            log.exception("Warmup failed: %s", str(e))

    warmup_task = asyncio.create_task(warmup(), name="warmup")

    # Wait for either a signal or API crash
    await stop_event.wait()

    # Cleanup
    log.info("Stopping dependencies...")

    if mcp_launcher is not None:
        try:
            await mcp_launcher.stop()
        except Exception:
            pass

    await vllm_launcher.stop()

    server.should_exit = True
    try:
        await api_task
    except Exception:
        pass

    if not warmup_task.done():
        warmup_task.cancel()

    log.info("Shutdown complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
