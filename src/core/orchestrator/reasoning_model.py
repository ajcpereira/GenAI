import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.orchestrator.prompt import SYSTEM_POLICY, build_reasoning_prompt
from core.orchestrator.models import ExecutionPlan, ReasoningDecision

class ReasoningModel:
    def __init__(self, config: dict):
        self.cfg = config["orchestrator"]
        self.allowed_topics_default = self.cfg["context_policy"]["allowed_topics"]
        self.denied_topics_default = self.cfg["context_policy"]["denied_topics"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        disable_via_env = bool(self.cfg["reasoning"].get("disable_model_via_env", True))
        self.disabled = (os.getenv("ORCH_DISABLE_MODEL", "1") == "1") if disable_via_env else False

        if not self.disabled:
            self._load()

    def _load(self) -> None:
        path = self.cfg["model"]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _looks_like_politics(self, text: str) -> bool:
        lowered = text.lower()
        markers = ["politic", "election", "parliament", "president", "prime minister", "partido", "eleições", "política"]
        return any(m in lowered for m in markers)

    def _deterministic_gate(self, user_prompt: str, allowed: List[str], denied: List[str]) -> ReasoningDecision:
        if self._looks_like_politics(user_prompt):
            return ReasoningDecision(False, "Request is out of scope: politics/elections are not allowed in this context.",
                                    ExecutionPlan(False, False, False, []))

        if allowed:
            if not any(t.lower().replace("-", " ") in user_prompt.lower() for t in allowed):
                return ReasoningDecision(False, "Request is out of scope: does not match allowed topics for this deployment.",
                                        ExecutionPlan(False, False, False, []))

        return ReasoningDecision(True, "", ExecutionPlan(True, False, False, []))

    def _extract_json(self, text: str) -> Optional[str]:
        start = text.rfind("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start:end+1]

    def decide(self, user_prompt: str, context_override: Optional[Dict[str, Any]] = None) -> ReasoningDecision:
        allowed = (context_override or {}).get("allowed_topics") or self.allowed_topics_default
        denied = (context_override or {}).get("denied_topics") or self.denied_topics_default

        if "politics" not in [d.lower() for d in denied]:
            denied = list(denied) + ["politics"]

        if self.disabled or self.model is None or self.tokenizer is None:
            return self._deterministic_gate(user_prompt, allowed, denied)

        system = SYSTEM_POLICY
        user = build_reasoning_prompt(user_prompt=user_prompt, allowed_topics=allowed, denied_topics=denied)
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        json_text = self._extract_json(text)
        if not json_text:
            return ReasoningDecision(False, "Orchestrator reasoning output invalid (no JSON). Failing closed.",
                                    ExecutionPlan(False, False, False, []))
        try:
            obj = json.loads(json_text)
            plan = obj.get("execution_plan") or {}
            decision = ReasoningDecision(
                within_context=bool(obj.get("within_context")),
                out_of_scope_reason=str(obj.get("out_of_scope_reason") or ""),
                execution_plan=ExecutionPlan(
                    use_llm=bool(plan.get("use_llm")),
                    use_rag=bool(plan.get("use_rag")),
                    use_mcp=bool(plan.get("use_mcp")),
                    tools=list(plan.get("tools") or []),
                ),
            )
        except Exception:
            return ReasoningDecision(False, "Orchestrator reasoning output invalid (JSON parse). Failing closed.",
                                    ExecutionPlan(False, False, False, []))

        if self._looks_like_politics(user_prompt):
            return ReasoningDecision(False, "Request is out of scope: politics/elections are not allowed in this context.",
                                    ExecutionPlan(False, False, False, []))

        return decision
