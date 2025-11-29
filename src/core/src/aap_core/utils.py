import re


def remove_thinking(response: str) -> str:
    """A snippet to remove thinking
    Note on the model Qwen 3: the openning <think> tag is missing. So we need to manually add it to beginning of the response

    Args:
        response (str): raw response

    Returns:
        str: response without thinking
    """
    if "</think>" in response and "<think>" not in response:
        response = "<think>" + response
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
