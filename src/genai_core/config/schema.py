from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, PositiveInt


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class LogRotationConfig(BaseModel):
    type: Literal["size", "time", "none"] = "size"
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    backup_count: int = Field(default=5, ge=0)
    when: str = "midnight"
    interval: PositiveInt = 1


class LoggingConfig(BaseModel):
    level: str = "INFO"
    core_log_file: Optional[str] = "./logs/core.log"
    rotation: LogRotationConfig = Field(default_factory=LogRotationConfig)


class VLLMConfig(BaseModel):
    model_path: str
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 8001
    python_bin: Optional[str] = None
    log_file: Optional[str] = "./logs/vllm.log"
    startup_timeout_s: int = 180
    extra_args: List[str] = Field(default_factory=list)


class OrchestratorConfig(BaseModel):
    vllm_base_url: str = "http://127.0.0.1:8001"
    model: str = "local-model"
    request_timeout_s: int = 120
    reserved_output_tokens: int = 128
    max_tokens_cap: int = 256
    # Optional chat template for prompt rendering (provider/model specific)
    chat_template_path: Optional[str] = "config/chat_templates/mistral_inst.jinja"
    # Lightweight session memory (Phase 1): number of user+assistant turns kept.
    history_max_turns: int = Field(default=6, ge=0, le=50)


class MCPConfigModel(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8765"
    # Web search tool id on the MCP server
    tool_name_web_search: str = "web_search"
    top_k: int = 5
    timeout_s: int = 10


class ToolsConfig(BaseModel):
    mcp: MCPConfigModel = Field(default_factory=MCPConfigModel)


class RagConfig(BaseModel):
    enabled: bool = False


class AppConfig(BaseModel):
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    vllm: VLLMConfig
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    rag: RagConfig = Field(default_factory=RagConfig)

    # Preserve unknown keys for forward compatibility if needed.
    extra: Dict[str, Any] = Field(default_factory=dict, exclude=True)
