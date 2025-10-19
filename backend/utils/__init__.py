from .course_utils import (
    detect_course_code,
    get_course_search_url,
    url_contains_course,
    title_starts_with_course,
    extract_course_fallbacks,
)
from .logging_utils import ensure_chat_log_file, log_chat_interaction
from .program_utils import (
    match_program_alias,
    same_program_family,
    build_program_index,
)

__all__ = [
    "detect_course_code",
    "get_course_search_url",
    "url_contains_course",
    "title_starts_with_course",
    "extract_course_fallbacks",
    "ensure_chat_log_file",
    "log_chat_interaction",
    "match_program_alias",
    "same_program_family",
    "build_program_index",
]