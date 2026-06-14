from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate arithmetic expressions safely"""

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Invalid calculation"