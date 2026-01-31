from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenBudget:
    max_context_tokens: int
    reserved_output_tokens: int
    target_prompt_tokens: int
    chars_per_token: float
    safety_factor: float


class TokenEstimator:
    def __init__(self, budget: TokenBudget):
        self.budget = budget

    def estimate_tokens(self, text: str) -> int:
        t = text or ""
        approx = int((len(t) / max(self.budget.chars_per_token, 1e-6)) * self.budget.safety_factor)
        return max(0, approx)

    def fits(self, prompt_text: str) -> bool:
        return self.estimate_tokens(prompt_text) <= self.budget.target_prompt_tokens
