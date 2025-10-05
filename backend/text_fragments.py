# text_fragments.py
# Build reliable Google-style Text Fragment URLs for highlighting:
# https://page#:~:text=...
# NOTE: We normalize *whitespace only* so the fragment stays a literal substring.

import re
import urllib.parse
from typing import Optional


_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9']+")


def _normalize_spaces_only(s: str) -> str:
    """Normalize whitespace only (do not change quotes/punctuation)."""
    if not s:
        return ""
    s = s.replace("\u00A0", " ")  # NBSP -> space
    s = _WS.sub(" ", s).strip()  # collapse whitespace
    return s


def _enc_spaces_only(s: str) -> str:
    return urllib.parse.quote(_normalize_spaces_only(s))


def build_text_fragment_url(
    base_url: str,
    *,
    text: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Return URL with a text fragment:
      #:~:text=[prefix-,]text[,-suffix]
    - Skips PDFs
    - Ensures fragment is the only hash at the end
    - Uses whitespace-only normalization for literal matching on the page
    """
    if not base_url:
        return base_url

    low = base_url.lower()
    if low.endswith(".pdf") or "content=pdf" in low:
        return base_url  # text fragments don't work on PDFs

    # text fragment must be the last (and only) hash
    stripped = base_url.split("#")[0]

    parts = []
    if prefix:
        parts.append(_enc_spaces_only(prefix) + "-,")
    parts.append(_enc_spaces_only(text))
    if suffix:
        parts.append(",-" + _enc_spaces_only(suffix))

    return f"{stripped}#:~:text={''.join(parts)}"


def choose_snippet(
    passage: str,
    hint: Optional[str] = None,
    max_chars: int = 140
) -> str:
    """
    Choose a compact, on-page snippet
    (prefer a sentence overlapping hint words).
    Do NOT add ellipses—text fragments must be exact substrings.
    """
    if not passage:
        return ""
    p = _normalize_spaces_only(passage)

    # split into sentences (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", p)
    sentences = [s for s in sentences if s]

    if hint:
        hw = {
            w.lower() for w in _WORD.findall(_normalize_spaces_only(hint))
            if len(w) > 2
        }
        
        def score(s: str) -> int:
            return len(hw.intersection({w.lower() for w in _WORD.findall(s)}))
        sentences.sort(key=score, reverse=True)

    s = sentences[0] if sentences else p
    if len(s) > max_chars:
        # trim to word boundary (no ellipsis)
        cut = s[:max_chars]
        sp = cut.rfind(" ")
        s = cut if sp == -1 else cut[:sp]
    return s


def is_synthetic_label(passage: str) -> bool:
    """
    Heuristic: skip fragments for synthetic items like "Links: <label>".
    Those strings don't appear verbatim on the destination page.
    """
    if passage is None:
        return False
    p = _normalize_spaces_only(passage)
    return p.lower().startswith("links:")
