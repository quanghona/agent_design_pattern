from typing import Union

from llama_index.core.callbacks.token_counting import get_tokens_from_response
from llama_index.core.llms import ChatResponse, CompletionResponse

from core.src.aap_core.types import TokenUsage


def token_from_response(usage: Union[CompletionResponse, ChatResponse]) -> TokenUsage:
    """
    Convert llamaindex's response object to a TokenUsage object.

    Args:
        usage (Union[CompletionResponse, ChatResponse]): The response object to convert.

    Returns:
        TokenUsage: The converted object.
    """
    input_tokens, output_tokens = get_tokens_from_response(usage)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
