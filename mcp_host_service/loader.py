# mcp_host_service/loader.py
import importlib
import inspect
import logging
import pkgutil
from typing import Type

from mcp_host_service.registry import ToolRegistry
from mcp_host_service.tool_types import Tool, ToolSpec

logger = logging.getLogger("mcp.loader")


def _is_tool_class(obj) -> bool:
    # Tool protocol shape: class with `spec: ToolSpec` and async `run(...)`
    if not inspect.isclass(obj):
        return False
    spec = getattr(obj, "spec", None)
    if not isinstance(spec, ToolSpec):
        return False
    run = getattr(obj, "run", None)
    return callable(run) and inspect.iscoroutinefunction(run)


def load_tools(registry: ToolRegistry) -> None:
    """
    Dynamically discover tools in the package `mcp_host_service.tools`.

    Any class in a module under that package that has:
      - `spec` as a ToolSpec instance
      - `async def run(self, inputs: dict) -> dict`
    will be instantiated (no-args ctor) and registered.
    """
    pkg_name = "mcp_host_service.tools"
    pkg = importlib.import_module(pkg_name)

    discovered = 0
    registered = 0

    for modinfo in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        discovered += 1
        mod_name = modinfo.name
        try:
            module = importlib.import_module(mod_name)
        except Exception as e:
            logger.exception("tool_module_import_failed", extra={"module": mod_name, "error": str(e)})
            continue

        for _, obj in inspect.getmembers(module):
            if not _is_tool_class(obj):
                continue

            tool_cls: Type[Tool] = obj
            try:
                tool = tool_cls()  # type: ignore[call-arg]
                registry.register(tool)
                registered += 1
            except Exception as e:
                logger.exception(
                    "tool_register_failed",
                    extra={"module": mod_name, "tool_class": getattr(tool_cls, "__name__", "unknown"), "error": str(e)},
                )

    logger.info("tools_loaded", extra={"modules_scanned": discovered, "tools_registered": registered})
