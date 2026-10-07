from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple

if TYPE_CHECKING:
    from .types import ContentType, MediaReference

# Magic-byte signatures used to sniff MIME types without external dependencies.
# Image formats are complete for the current scope; audio/video/document
# signatures get appended here as those modalities are wired up (the transport
# classification below is already modality-agnostic).
_MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _sniff_mime_type(data: bytes) -> str | None:
    """Detect MIME type from magic bytes. Returns None when unknown."""
    for signature, mime_type in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime_type
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def resolve_media(
    media: Sequence[Tuple[ContentType, str]] | None,
) -> List[MediaReference]:
    """Resolve media references carried in AgentMessage fields to MediaRef objects.

    Transport classification is modality-agnostic: any ContentType value can be
    resolved today. Each (content_type, value) pair is classified by its value:
    - "http://" or "https://" prefix -> kind="url", passed through as-is.
    - "data:" prefix -> kind="base64", MIME type and payload extracted from the URI.
    - "file://" prefix or anything else -> treated as a local path; the file is read
      and base64-encoded, with MIME type sniffed from magic bytes (falling back to
      mimetypes, then to "application/octet-stream").

    A local path that does not exist raises FileNotFoundError.

    Args:
        media: List of (content_type, value) tuples, e.g. a filtered slice of
            AgentMessage.query_media.

    Returns:
        List[MediaRef]: Resolved references, each tagged with its content_type,
        order preserved. Empty list if media is empty.
    """
    refs: List[MediaReference] = []
    for content_type, value in media or []:
        if value.startswith(("http://", "https://")):
            mime_type = mimetypes.guess_type(value.split("?")[0])[0]
            refs.append(
                {
                    "content_type": content_type,
                    "kind": "url",
                    "value": value,
                    "mime_type": mime_type or "application/octet-stream",
                }
            )
        elif value.startswith("data:"):
            header, _, payload = value.partition(",")
            mime_type = header[len("data:") :].split(";")[0]
            refs.append(
                {
                    "content_type": content_type,
                    "kind": "base64",
                    "value": payload,
                    "mime_type": mime_type or "application/octet-stream",
                }
            )
        else:
            path = Path(value.removeprefix("file://"))
            data = path.read_bytes()
            mime_type = (
                _sniff_mime_type(data)
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            refs.append(
                {
                    "content_type": content_type,
                    "kind": "base64",
                    "value": base64.b64encode(data).decode("ascii"),
                    "mime_type": mime_type,
                }
            )
    return refs


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
