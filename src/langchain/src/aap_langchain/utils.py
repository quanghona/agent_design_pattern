import logging
import re
from collections.abc import Iterable
from typing import Any, Dict, Tuple

from aap_core.types import ContentType, MediaReference, TokenUsage
from langchain_core.messages.ai import UsageMetadata

logger = logging.getLogger(__name__)

# ModelProfile input-modality keys, mapped onto the unified ContentType.
# A profile is a total=False TypedDict, so any key may be absent; absent keys
# stay undecidable rather than being reported as unsupported.
_PROFILE_INPUT_KEYS: Dict[ContentType, str] = {
    "image": "image_inputs",
    "audio": "audio_inputs",
    "video": "video_inputs",
    "document": "pdf_inputs",
}

# Hugging Face repo ids are the only identifiers that resolve to a model card.
_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_REPO_ID_PREFIXES = ("huggingface:", "hf:")

# Modality keywords, matched against the input side of an HF task tag.
_MODALITY_TAG_KEYWORDS: Dict[ContentType, Tuple[str, ...]] = {
    "image": ("image", "visual", "vision", "chart", "ocr"),
    "audio": ("audio", "speech", "voice"),
    "video": ("video", "frame"),
    "document": ("document", "pdf"),
}

# The missing-extra warning should surface once per process, not per lookup.
_HUB_UNAVAILABLE_LOGGED = False


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


def media_ref_to_content_block(ref: MediaReference) -> Dict[str, Any]:
    """
    Convert a resolved MediaRef to a langchain content block.

    The mapping is modality-agnostic for image and audio, whose base64/url
    block shapes are identical in langchain-core 1.x. Document and video
    blocks use different fields and will need a branch when wired up.

    Args:
        ref (MediaRef): The resolved media reference from BaseLLMChain._media_parts.

    Returns:
        Dict[str, Any]: A langchain content block ready to embed in a message's content list.
    """
    if ref["kind"] == "url":
        return {"type": ref["content_type"], "url": ref["value"]}
    return {
        "type": ref["content_type"],
        "base64": ref["value"],
        "mime_type": ref["mime_type"],
    }


def profile_to_capabilities(profile: Any) -> Dict[ContentType, bool]:
    """Map a langchain model profile onto per-modality input capabilities.

    The profile is the provider-shipped capability table (see the LangChain
    model profiles guide); it is authoritative per model revision, which is
    what makes it preferable to model-family name matching. Only keys actually
    present in the profile produce an opinion.

    Args:
        profile: A ModelProfile mapping, or None/anything non-dict.

    Returns:
        Dict[ContentType, bool]: Modalities the profile asserts an opinion on.
    """
    if not isinstance(profile, dict):
        return {}
    capabilities: Dict[ContentType, bool] = {}
    for content_type, key in _PROFILE_INPUT_KEYS.items():
        value = profile.get(key)
        if isinstance(value, bool):
            capabilities[content_type] = value
    return capabilities


def task_tags_to_capabilities(
    task_tags: Iterable[str | None],
) -> Dict[ContentType, bool]:
    """Map the task tags a Hugging Face repo declares onto input modalities.

    Only the input side of each tag counts, so "text-to-speech" does not imply
    audio input. When a repo declares at least one task, the tags are read as a
    complete statement of what it accepts: a modality no tag mentions reports
    False rather than staying unknown. That is what separates variants of one
    family, e.g. google/gemma-3n-e4b-it (tagged audio-text-to-text) from
    google/gemma-3-27b-it (tagged image-text-to-text only). A repo published
    under an incomplete tag set will therefore under-report; media_support is
    the override. With no usable tag at all the result is empty, i.e.
    undecidable, which keeps auto mode fail-loud.

    Args:
        task_tags: Declared task identifiers, e.g. "image-text-to-text".

    Returns:
        Dict[ContentType, bool]: One entry per gated modality, or empty when the
        repo declares no usable task.
    """
    tags = [tag.lower() for tag in task_tags if isinstance(tag, str) and tag.strip()]
    if not tags:
        return {}
    if any(tag == "any-to-any" for tag in tags):
        return {modality: True for modality in _MODALITY_TAG_KEYWORDS}
    # Document/PDF support is not expressed by the task-tag grammar, so it is
    # only reported when positively matched; never denied from absence.
    capabilities: Dict[ContentType, bool] = {
        modality: False for modality in ("image", "audio", "video")
    }
    for tag in tags:
        input_side = tag.split("-to-")[0]
        for modality, keywords in _MODALITY_TAG_KEYWORDS.items():
            if any(keyword in input_side for keyword in keywords):
                capabilities[modality] = True
    return capabilities


def extract_repo_id(candidates: Iterable[str | None]) -> str | None:
    """Pick the first candidate that is a Hugging Face repo id (namespace/name).

    Provider model names are not repo ids ("gpt-4o", "gemma3:12b"), so they are
    rejected here rather than looked up and 404'd. Qualifiers are stripped so
    "org/repo:q8_0" or "org/repo@rev" still resolve.

    Args:
        candidates: Identifier strings in priority order; non-strings ignored.

    Returns:
        str | None: The repo id, or None if no candidate looks like one.
    """
    for raw in candidates:
        if not isinstance(raw, str) or not raw.strip():
            continue
        name = raw.strip()
        lowered = name.lower()
        for prefix in _REPO_ID_PREFIXES:
            if lowered.startswith(prefix):
                name = name[len(prefix) :]
                break
        name = re.split(r"[@:]", name, maxsplit=1)[0]
        if _REPO_ID_PATTERN.match(name):
            return name
    return None


def huggingface_capabilities(repo_id: str) -> Dict[ContentType, bool]:
    """Look up a model's input modalities from its Hugging Face model card.

    Performs one network request; callers are expected to cache the result.
    Requires the "hf" extra; without it, or when offline, or for a private or
    renamed repo, the lookup reports undecidable so auto mode keeps the
    fail-loud behavior instead of breaking the request path.

    Args:
        repo_id (str): The Hugging Face repo id, e.g. "google/gemma-3-4b-it".

    Returns:
        Dict[ContentType, bool]: Capabilities read from the repo's declared
        tasks, possibly empty.
    """
    global _HUB_UNAVAILABLE_LOGGED
    try:
        from huggingface_hub import model_info  # type: ignore[import-not-found]
    except ImportError:
        if not _HUB_UNAVAILABLE_LOGGED:
            _HUB_UNAVAILABLE_LOGGED = True
            logger.warning(
                "huggingface-hub is not installed, so Hugging Face capability "
                'detection is off. Install the "hf" extra of aap_langchain to '
                "enable it."
            )
        return {}
    try:
        info = model_info(repo_id)
    except Exception as exc:  # noqa: BLE001 - detection must never break invoke()
        logger.debug("Hugging Face metadata lookup failed for %s: %s", repo_id, exc)
        return {}
    tags = [getattr(info, "pipeline_tag", None)]
    tags += [tag for tag in getattr(info, "tags", None) or [] if "-to-" in tag]
    return task_tags_to_capabilities(tags)
