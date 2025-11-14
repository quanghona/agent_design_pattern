import re


def remove_thinking(response: str) -> str:
    # Snippet to remove thinking
    if "</think>" in response and "<think>" not in response:
        response = "<think>" + response
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
