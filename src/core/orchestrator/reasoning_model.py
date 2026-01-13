class ReasoningModel:
    def __init__(self, config):
        self.cfg = config["orchestrator"]

    def decide(self, prompt):
        return {
            "use_llm": True,
            "use_rag": False,
            "use_mcp": False,
            "tools": []
        }
