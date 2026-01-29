from aap_core.types import TokenUsage
from langchain_core.messages.ai import UsageMetadata


def token_from_response(usage: UsageMetadata) -> TokenUsage:
    """
    Convert langchain's usage object to a TokenUsage object.

    Args:
        usage (UsageMetadata): The UsageMetadata object to convert.

    Returns:
        TokenUsage: The converted object.
    """
    return TokenUsage(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
    )
