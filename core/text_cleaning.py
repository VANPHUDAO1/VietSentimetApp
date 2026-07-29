"""Raw text cleaning pipeline used before normalization and segmentation."""

import re
import unicodedata
from functools import lru_cache

import emoji
import pandas as pd
from underthesea import text_normalize

from config.settings import TEENCODE_FILE

EMOJI_LEXICON = {
    "😊": "mặt cười hạnh phúc",
    "😄": "mặt cười tươi",
    "😁": "mặt cười rạng rỡ",
    "🙂": "mặt cười nhẹ",
    "😀": "mặt cười lớn",
    "😃": "mặt cười vui vẻ",
    "🤗": "ôm ấm áp",
    "😂": "cười lăn lộn",
    "🤣": "cười ngặt nghẽo",
    "😆": "cười lớn",
    "😅": "cười ngại ngùng",
    "🤭": "cười che miệng",
    "❤": "trái tim đỏ",
    "❤️": "trái tim đỏ",
    "🧡": "trái tim cam",
    "💛": "trái tim vàng",
    "💚": "trái tim xanh lá",
    "💙": "trái tim xanh dương",
    "💜": "trái tim tím",
    "🤍": "trái tim trắng",
    "🖤": "trái tim đen",
    "🤎": "trái tim nâu",
    "💕": "hai trái tim",
    "💞": "trái tim xoay",
    "💓": "trái tim đập",
    "💗": "trái tim lớn dần",
    "💘": "trái tim tên bắn",
    "💝": "trái tim hộp quà",
    "💟": "trái tim trang trí",
    "♥": "trái tim",
    "😍": "mặt mắt hình tim",
    "🥰": "mặt cười yêu thương",
    "😘": "thổi hôn",
    "😚": "hôn mặt",
    "😙": "hôn cười",
    "👍": "ngón tay cái lên",
    "👏": "vỗ tay",
    "🙌": "hai tay giơ cao",
    "✊": "nắm đấm",
    "👊": "đấm nhẹ",
    "💪": "cơ bắp mạnh mẽ",
    "🤝": "bắt tay",
    "🙏": "cảm ơn hoặc cầu nguyện",
    "🔥": "rất hot",
    "⭐": "ngôi sao",
    "🌟": "ngôi sao sáng",
    "✨": "lấp lánh",
    "💥": "bùng nổ",
    "⚡": "sét đánh",
    "🏆": "cúp chiến thắng",
    "🥇": "huy chương vàng",
    "😢": "mặt buồn khóc",
    "😭": "khóc lớn",
    "😔": "mặt buồn bã",
    "😞": "mặt thất vọng",
    "😟": "mặt lo lắng",
    "🥺": "mặt van xin tội nghiệp",
    "😿": "mèo buồn",
    "👎": "ngón tay cái xuống",
    "😠": "mặt tức giận",
    "😡": "mặt rất giận",
    "🤬": "mặt chửi thề",
    "😤": "mặt bực bội",
    "😮": "mặt ngạc nhiên",
    "😲": "mặt kinh ngạc",
    "🤯": "đầu nổ tung",
    "😯": "mặt ngỡ ngàng",
    "😦": "mặt lo sợ",
    "😧": "mặt hoảng hốt",
    "🤔": "mặt suy nghĩ",
    "😐": "mặt bình thường",
    "😑": "mặt thờ ơ",
    "🙄": "mặt đảo mắt",
    "😏": "mặt mỉa mai",
    "😒": "mặt không hài lòng",
    "🥱": "mặt ngáp",
    "😴": "mặt đang ngủ",
    "🤢": "mặt buồn nôn",
    "🤮": "mặt nôn mửa",
    "😷": "mặt đeo khẩu trang",
    "🤒": "mặt bị sốt",
    "🥴": "mặt choáng váng",
    "😵": "mặt chóng mặt",
}

EMOTICON_DICT = {
    ":)": "vui",
    ":-)": "vui",
    ":))": "cười",
    ":)))": "cười lớn",
    ":))))": "cười lớn",
    "=)": "vui",
    "^_^": "vui",
    ":D": "cười lớn",
    ":-D": "cười lớn",
    "XD": "cười lớn",
    "xD": "cười lớn",
    "=D": "cười lớn",
    ":(": "buồn",
    ":-(": "buồn",
    ":'(": "khóc",
    "T_T": "khóc",
    "TT": "khóc",
    "ㅠㅠ": "khóc",
    ";_;": "khóc",
    ">:(": "tức giận",
    ":-@": "tức giận",
    "D:<": "tức giận",
    ">:O": "tức giận",
    ":O": "ngạc nhiên",
    ":-O": "ngạc nhiên",
    "o_O": "ngạc nhiên",
    "O_O": "ngạc nhiên",
    "O.o": "ngạc nhiên",
    ":/": "nghi ngờ",
    ":-/": "nghi ngờ",
    ":|": "trung tính",
    "-_-": "chán",
    "-.-": "chán",
    ";)": "nháy mắt",
    ";-)": "nháy mắt",
    ";D": "nháy mắt vui",
    "<3": "yêu",
    "♥": "yêu",
    ":*": "hôn",
    ":-*": "hôn",
    ":P": "trêu",
    ":-P": "trêu",
    "xP": "trêu",
    "XP": "trêu",
    ":p": "trêu",
    "._.": "thất vọng",
    "T.T": "khóc",
    "orz": "thất vọng",
}


URL_EMAIL_PATTERN = re.compile(
    r"\b[\w\.-]+@[\w\.-]+\.\w+\b"  # Email
    r"|(?:https?://|ftp://|www\.)\S+"  # URL co protocol hoac www
    r"|\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/\S*)?",  # Domain khong co protocol
    flags=re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_teencode_dict() -> dict:
    """Load the teencode dictionary once and cache the result."""
    if not TEENCODE_FILE.exists():
        return {}

    df_teencode = pd.read_csv(
        TEENCODE_FILE,
        sep="\t",
        header=None,
        names=["teencode", "meaning"],
        encoding="utf-8-sig",
    )
    return {
        rf"\b{re.escape(row.teencode)}\b": row.meaning
        for _, row in df_teencode.iterrows()
        if pd.notna(row.teencode) and pd.notna(row.meaning)
    }


def remove_url(text: str) -> str:
    """Remove URLs, emails, and domains from text."""
    text = URL_EMAIL_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def replace_emoji_with_text(text: str, emoji_dict: dict = EMOJI_LEXICON) -> str:
    """Replace known emojis with Vietnamese text, remove unknown emojis."""
    result = []
    for token in emoji.analyze(text, non_emoji=True):
        val = token.value
        if hasattr(val, "emoji"):
            if val.emoji in emoji_dict:
                result.append(f" {emoji_dict[val.emoji]} ")
        else:
            result.append(val)
    return re.sub(r"\s+", " ", "".join(result)).strip()


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC."""
    return unicodedata.normalize("NFC", text)


def remove_timestamp(text: str) -> str:
    """Remove timestamp patterns like mm:ss or hh:mm:ss."""
    text = re.sub(r"\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    """Collapse repeated punctuation marks."""
    return re.sub(r"([.,!?;:])\1+", r"\1", text)


def normalize_repeated_chars(text: str) -> str:
    """Reduce repeated characters to two occurrences."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def replace_teencode(text: str, teen_dict: dict) -> str:
    """Replace teencode tokens with their normalized equivalents."""
    for pattern, replacement in teen_dict.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_special_chars(text: str, emoticon_dict: dict = EMOTICON_DICT) -> str:
    """Remove leftover emoticons and other special characters."""
    emoticon_pattern = "|".join(
        re.escape(k) for k in sorted(emoticon_dict, key=len, reverse=True)
    )
    text = re.sub(emoticon_pattern, " ", text)
    text = re.sub(
        r"[^\w\sàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
        r"ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ.,!?]",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text accents using underthesea."""
    return text_normalize(text)


def clean_raw_text(text: str) -> str:
    """Run the raw text cleaning pipeline in fixed order."""
    if not isinstance(text, str):
        return ""

    teen_dict = _load_teencode_dict()

    text = remove_url(text)
    text = remove_html_tags(text)
    text = replace_emoji_with_text(text, EMOJI_LEXICON)
    text = normalize_unicode(text)
    text = remove_timestamp(text)
    text = normalize_punctuation(text)
    text = normalize_repeated_chars(text)
    text = to_lowercase(text)
    text = replace_teencode(text, teen_dict)
    text = remove_special_chars(text, EMOTICON_DICT)
    text = normalize_whitespace(text)
    text = normalize_vietnamese(text)
    return text
