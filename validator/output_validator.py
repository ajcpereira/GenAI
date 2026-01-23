# validator/output_validator.py
from typing import Any, Dict, List

from utils.common import validate_json


class OutputValidator:
    def __init__(self, contract_bundle: Dict[str, Any]):
        self.contract_bundle = contract_bundle
        self.step_exec_schema = contract_bundle["schemas"]["StepExecution"]

    def validate(self, execution_result: Dict[str, Any]) -> None:
        if "steps_executed" not in execution_result:
            raise ValueError("Missing steps_executed in execution result")

        steps: List[Dict[str, Any]] = execution_result.get("steps_executed") or []
        for s in steps:
            validate_json(self.step_exec_schema, s, bundle=self.contract_bundle)
