# mcp_host_service/tools/math_eval.py
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from mcp_host_service.tool_types import ToolSpec


Number = Union[int, float]


class _SafeMathError(ValueError):
    pass


class _SafeMathEvaluator(ast.NodeVisitor):
    """
    Deterministic, side-effect-free math expression evaluator.

    Allowed:
      - Literals: int, float
      - Operators: +, -, *, /, %, **, unary +/-
      - Parentheses via AST structure
      - Names: pi, e
      - Calls: sqrt, abs, round, floor, ceil, min, max
    Disallowed:
      - Variables (other than allowed constants)
      - Attribute access (math.sqrt)
      - Subscripting, comprehensions, lambdas
      - Strings, bytes, f-strings
      - Any statement nodes
    """

    def __init__(
        self,
        *,
        max_expr_len: int = 512,
        max_nodes: int = 200,
        max_pow_abs_exponent: int = 10_000,
    ) -> None:
        self.max_expr_len = max_expr_len
        self.max_nodes = max_nodes
        self.max_pow_abs_exponent = max_pow_abs_exponent
        self._node_count = 0

        self._consts: Dict[str, Number] = {
            "pi": math.pi,
            "e": math.e,
        }
        self._funcs: Dict[str, Any] = {
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
            "min": min,
            "max": max,
        }

    def eval(self, expression: str) -> Number:
        expr = (expression or "").strip()
        if not expr:
            raise _SafeMathError("Empty expression.")
        if len(expr) > self.max_expr_len:
            raise _SafeMathError(f"Expression too long (max {self.max_expr_len} chars).")

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise _SafeMathError(f"Invalid expression syntax: {e.msg}") from e

        self._node_count = 0
        value = self.visit(tree.body)
        if not isinstance(value, (int, float)):
            raise _SafeMathError("Expression did not evaluate to a number.")
        if isinstance(value, bool):
            raise _SafeMathError("Boolean results are not allowed.")
        return value

    # --- Limits / counting ---
    def generic_visit(self, node: ast.AST):
        self._node_count += 1
        if self._node_count > self.max_nodes:
            raise _SafeMathError(f"Expression too complex (max {self.max_nodes} nodes).")
        return super().generic_visit(node)

    # --- Literals / names ---
    def visit_Constant(self, node: ast.Constant) -> Number:
        self._node_count += 1
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise _SafeMathError(f"Unsupported literal type: {type(node.value).__name__}")

    def visit_Name(self, node: ast.Name) -> Number:
        self._node_count += 1
        name = str(node.id)
        if name in self._consts:
            return self._consts[name]
        raise _SafeMathError(f"Unknown identifier: {name}")

    # --- Operators ---
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Number:
        self._node_count += 1
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise _SafeMathError("Unsupported unary operator.")

    def visit_BinOp(self, node: ast.BinOp) -> Number:
        self._node_count += 1
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            # basic DoS guard on exponent magnitude when exponent is integral-like
            if isinstance(right, int) and abs(right) > self.max_pow_abs_exponent:
                raise _SafeMathError(f"Exponent too large (abs max {self.max_pow_abs_exponent}).")
            # if float exponent, still allow, but node count/len limits apply
            return left ** right

        raise _SafeMathError("Unsupported binary operator.")

    # --- Calls ---
    def visit_Call(self, node: ast.Call) -> Number:
        self._node_count += 1

        # Only allow direct function names (no attributes, no lambdas, etc.)
        if not isinstance(node.func, ast.Name):
            raise _SafeMathError("Only direct function calls are allowed (e.g., sqrt(9)).")

        fname = str(node.func.id)
        fn = self._funcs.get(fname)
        if fn is None:
            raise _SafeMathError(f"Unsupported function: {fname}")

        # No keyword arguments for simplicity/safety
        if node.keywords:
            raise _SafeMathError("Keyword arguments are not allowed.")

        args = [self.visit(a) for a in node.args]
        try:
            out = fn(*args)
        except Exception as e:
            raise _SafeMathError(f"Math error while evaluating {fname}(...): {str(e)}") from e

        if isinstance(out, (int, float)) and not isinstance(out, bool):
            return out
        raise _SafeMathError("Function call did not return a number.")

    # --- Explicitly block things we do not support ---
    def visit_Attribute(self, node: ast.Attribute):
        raise _SafeMathError("Attribute access is not allowed.")

    def visit_Subscript(self, node: ast.Subscript):
        raise _SafeMathError("Subscript access is not allowed.")

    def visit_List(self, node: ast.List):
        raise _SafeMathError("Lists are not allowed.")

    def visit_Tuple(self, node: ast.Tuple):
        raise _SafeMathError("Tuples are not allowed.")

    def visit_Dict(self, node: ast.Dict):
        raise _SafeMathError("Dicts are not allowed.")

    def visit_Compare(self, node: ast.Compare):
        raise _SafeMathError("Comparisons are not allowed.")

    def visit_BoolOp(self, node: ast.BoolOp):
        raise _SafeMathError("Boolean operations are not allowed.")

    def visit_IfExp(self, node: ast.IfExp):
        raise _SafeMathError("Conditional expressions are not allowed.")


class MathEvalTool:
    spec = ToolSpec(
        name="math.eval",
        description=(
            "Evaluate a pure mathematical expression deterministically and return its numeric result. "
            "Supports basic arithmetic (+, -, *, /, %, **), parentheses, constants (pi, e), and common functions "
            "(sqrt, abs, round, floor, ceil, min, max). "
            "Does NOT support variables (except pi/e), strings, imports, attribute access, or any side effects."
        ),
        input_schema={
            "type": "object",
            "required": ["expression"],
            "properties": {
                "expression": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {"type": ["number", "string"]},
            },
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        # Keep limits conservative to avoid abuse; adjust via code later if needed.
        self._evaluator = _SafeMathEvaluator(
            max_expr_len=512,
            max_nodes=200,
            max_pow_abs_exponent=10_000,
        )

    async def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        expr = inputs.get("expression")
        if expr is None:
            # Schema should prevent this, but keep defense-in-depth.
            raise _SafeMathError("Missing required field: expression")

        value = self._evaluator.eval(str(expr))

        # If the result is an int too large for safe JSON number handling in some clients,
        # return it as a string. Use 2**53-1 boundary (JS safe integer).
        if isinstance(value, int) and abs(value) > 9_007_199_254_740_991:
            return {"value": str(value)}

        return {"value": value}
