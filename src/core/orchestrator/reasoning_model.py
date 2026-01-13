class ReasoningModel:
    def __init__(self, config):
        self.allow_tools = config["orchestrator"]["reasoning"]["allow_tools"]
        self.allow_rag = config["orchestrator"]["reasoning"]["allow_rag"]

    def decide(self, prompt):
        return {
            "use_llm": True,
            "use_rag": False,
            "use_mcp": False,
            "tools": []
        }
