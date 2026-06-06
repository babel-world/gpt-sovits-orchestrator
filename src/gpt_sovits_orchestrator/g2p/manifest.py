from __future__ import annotations

import csv
import io
from collections.abc import Callable
from typing import Any

from gpt_sovits_orchestrator.config import (
    G2P_COLUMNS,
    PRIVACY_ALLOWED_LANGUAGE,
    PRIVACY_MIN_PROBABILITY,
)
from gpt_sovits_orchestrator.g2p.japanese import text_to_vits_phones
from gpt_sovits_orchestrator.g2p.normalize import normalize_japanese
from gpt_sovits_orchestrator.g2p.validation import find_unknown_symbols


def _privacy_gate(row: dict[str, str]) -> tuple[bool, str]:
    language = (row.get("language") or "").strip()
    if language != PRIVACY_ALLOWED_LANGUAGE:
        return False, "privacy_language_not_allowed"

    probability_raw = (row.get("probability") or "").strip()
    try:
        probability = float(probability_raw)
    except ValueError:
        return False, "privacy_probability_too_low"

    if probability <= PRIVACY_MIN_PROBABILITY:
        return False, "privacy_probability_too_low"

    return True, ""


def _append_g2p_columns(
    row: dict[str, str],
    *,
    norm_text: str = "",
    phones: str = "",
    phone_count: str = "",
    word2ph: str = "",
    status: str = "",
    error: str = "",
) -> dict[str, str]:
    row["norm_text"] = norm_text
    row["phones"] = phones
    row["phone_count"] = phone_count
    row["word2ph"] = word2ph
    row["status"] = status
    row["error"] = error
    return row


def _process_row(
    row: dict[str, Any],
    g2p_fn: Callable[[str], list[str]],
) -> dict[str, str]:
    output = {key: "" if value is None else str(value) for key, value in row.items()}

    allowed, privacy_error = _privacy_gate(output)
    if not allowed:
        return _append_g2p_columns(
            output,
            status="skip",
            error=privacy_error,
        )

    text = (output.get("text") or "").strip()
    if not text:
        return _append_g2p_columns(
            output,
            status="skip",
            error="empty_text",
        )

    try:
        norm_text = normalize_japanese(text)
        phones_list = text_to_vits_phones(norm_text, g2p_fn)
        unknown_symbols = find_unknown_symbols(phones_list)
        if unknown_symbols:
            return _append_g2p_columns(
                output,
                norm_text=norm_text,
                status="error",
                error=f"unknown: {unknown_symbols}",
            )

        return _append_g2p_columns(
            output,
            norm_text=norm_text,
            phones=" ".join(phones_list),
            phone_count=str(len(phones_list)),
            word2ph="None",
            status="ok",
        )
    except Exception as exc:
        return _append_g2p_columns(
            output,
            status="error",
            error=str(exc),
        )


def process_manifest_csv(
    csv_text: str,
    g2p_fn: Callable[[str], list[str]],
) -> str:
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header")

    fieldnames = list(reader.fieldnames)
    for column in G2P_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()

    for row in reader:
        writer.writerow(_process_row(row, g2p_fn))

    return output.getvalue()


def summarize_manifest_csv(csv_text: str) -> dict[str, int]:
    reader = csv.DictReader(io.StringIO(csv_text))
    summary = {"ok": 0, "skip": 0, "error": 0, "total": 0}

    for row in reader:
        summary["total"] += 1
        status = (row.get("status") or "").strip()
        if status in summary:
            summary[status] += 1

    return summary
