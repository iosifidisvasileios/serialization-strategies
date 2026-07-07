from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Optional

from .label_utils import label_to_key


MONEY_LABEL_KEYWORDS = {
    "amount", "balance", "commission", "cost", "fee", "gross", "net",
    "price", "rate", "subtotal", "tax", "total", "value",
}

DATE_LABEL_KEYWORDS = {
    "date", "start_date", "end_date", "effective_date", "expiration_date",
}


def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def compact_alnum(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def is_money_label(label: str) -> bool:
    key = label_to_key(label)
    return any(part in key for part in MONEY_LABEL_KEYWORDS)


def is_date_label(label: str) -> bool:
    key = label_to_key(label)
    return any(part in key for part in DATE_LABEL_KEYWORDS)


def strip_ocr_suffixes(value: str) -> str:
    """Remove common OCR/table artifacts observed in FCC invoices.

    Examples:
        $150.00P-2    -> $150.00
        $2,000.00P-1 -> $2,000.00
        900.00P-2    -> 900.00
    """
    value = str(value).strip()
    value = re.sub(r"(?i)p-\d+$", "", value)
    value = re.sub(r"(?i)p\d+$", "", value)
    return value.strip()


def extract_first_number_like(value: Any) -> Optional[str]:
    text = str(value)

    # Parenthesized negative: ($1,234.50)
    m = re.search(r"\(\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*\)", text)
    if m:
        return "-" + m.group(1)

    # Optional sign/currency, commas, decimal.
    m = re.search(r"[-+]?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", text)
    if m:
        prefix = "-" if m.group(0).strip().startswith("-") else ""
        return prefix + m.group(1)

    return None


def normalize_money(value: Any, keep_commas: bool = False) -> str:
    text = normalize_whitespace(value)
    text = strip_ocr_suffixes(text)
    number = extract_first_number_like(text)

    if number is None:
        cleaned = text.replace("$", "")
        cleaned = cleaned.replace(",", "" if not keep_commas else ",")
        cleaned = re.sub(r"(?i)p-\d+$", "", cleaned).strip()
        return cleaned

    if not keep_commas:
        number = number.replace(",", "")

    return number.strip()


def normalize_decimal_string(value: Any) -> str:
    text = normalize_money(value, keep_commas=False)
    try:
        dec = Decimal(text)
    except (InvalidOperation, ValueError):
        return text

    if "." in text:
        places = len(text.split(".", 1)[1])
        quant = Decimal("1." + ("0" * places))
        return str(dec.quantize(quant))

    return str(dec)


def normalize_date(value: Any) -> str:
    return normalize_whitespace(value).replace(" ", "")


def normalize_text_value(value: Any) -> str:
    text = normalize_whitespace(value)
    text = text.replace("‹", "<").replace("›", ">")
    text = text.replace("–", "-").replace("—", "-")
    return text.strip()


def normalize_for_matching(label: str, value: Any) -> str:
    if value is None:
        return ""
    if is_money_label(label):
        return normalize_money(value)
    if is_date_label(label):
        return compact_alnum(normalize_date(value))
    return compact_alnum(normalize_text_value(value))


def normalize_extracted_value(label: str, value: Any) -> str:
    if value is None:
        return ""
    if is_money_label(label):
        return normalize_money(value)
    if is_date_label(label):
        return normalize_date(value)
    return normalize_text_value(value)


def loose_value_match(label: str, gold_value: Any, token_value: Any) -> bool:
    """Compare gold annotation text to noisy OCR token text.

    Treat these as matches:
        gold="150.00", token="$150.00P-2"
        gold="17,410.00", token="$17,410.00"
    """
    gold_norm = normalize_for_matching(label, gold_value)
    token_norm = normalize_for_matching(label, token_value)

    if not gold_norm or not token_norm:
        return gold_norm == token_norm

    return gold_norm == token_norm or gold_norm in token_norm or token_norm in gold_norm


def normalize_value_pair(label: str, gold_value: Any, pred_value: Any) -> tuple[str, str]:
    return (
        normalize_extracted_value(label, gold_value),
        normalize_extracted_value(label, pred_value),
    )


def normalized_exact_match(label: str, gold_value: Any, pred_value: Any) -> bool:
    gold_norm, pred_norm = normalize_value_pair(label, gold_value, pred_value)
    return gold_norm == pred_norm


def json_safe_marker(value: Any) -> str:
    try:
        import json
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return repr(value)


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        marker = json_safe_marker(value)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(value)
    return out


def normalize_json_values(obj: Any, parent_label: Optional[str] = None) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_json_values(v, parent_label=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_json_values(v, parent_label=parent_label) for v in obj]
    if parent_label is None:
        return normalize_text_value(obj)
    return normalize_extracted_value(parent_label, obj)
