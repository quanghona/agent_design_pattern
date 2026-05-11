from aap_core.types import TokenUsage
from dspy import Prediction


def token_from_response(prediction: Prediction) -> TokenUsage:
    """
    Convert dspy's prediction object to a TokenUsage object.

    Args:
        predicted (Prediction): The prediction object to convert.

    Returns:
        TokenUsage: The converted object.
    """
    usage = prediction.get_lm_usage()
    if usage is not None and len(usage) > 0:
        usage = list(usage.values())[0]
        return TokenUsage(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )
    return TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
