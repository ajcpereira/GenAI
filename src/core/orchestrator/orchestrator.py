from .reasoning_model import ReasoningModel
from observability.logger import log

class Orchestrator:
    def __init__(self, config):
        self.reasoner = ReasoningModel(config)

    def plan(self, prompt):
        plan = self.reasoner.decide(prompt)
        log("execution_plan", plan=plan)
        return plan
