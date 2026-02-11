from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class GSM8KExample:
    example_id: str
    question: str
    answer: str
    gold_answer: str


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
DELIM_RE = re.compile(r"####\s*([^\n\r]+)")


def _canonicalize_number(raw: str) -> Optional[str]:
    cleaned = raw.strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(" ", "")
    if not cleaned:
        return None

    match = NUMBER_RE.search(cleaned)
    if not match:
        return None

    token = match.group(0)
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None

    if value == value.to_integral_value():
        return str(int(value))

    normalized = format(value.normalize(), "f")
    normalized = normalized.rstrip("0").rstrip(".")
    return normalized if normalized else "0"


def parse_final_answer(text: str) -> Optional[str]:
    delim_matches = DELIM_RE.findall(text)
    if delim_matches:
        parsed = _canonicalize_number(delim_matches[-1])
        if parsed is not None:
            return parsed

    fallback = None
    for match in NUMBER_RE.finditer(text):
        fallback = match.group(0)

    if fallback is None:
        return None
    return _canonicalize_number(fallback)


def exact_match_reward(predicted: Optional[str], gold: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted == gold else 0.0


def shaped_numeric_reward(predicted: Optional[str], gold: str, scale: float) -> float:
    if predicted is None:
        return 0.0
    if scale <= 0:
        raise ValueError("scale must be > 0")

    if predicted == gold:
        return 1.0

    try:
        pred_v = float(predicted)
        gold_v = float(gold)
    except ValueError:
        return 0.0

    distance = abs(pred_v - gold_v)
    return float(max(0.0, min(1.0, math.exp(-distance / scale))))


def build_prompt(question: str, best_context: Optional[str] = None) -> str:
    lines = [
        "Solve the following math word problem.",
        "Return the final answer on its own line as: #### <answer>",
        "",
        f"Question: {question}",
    ]

    if best_context:
        lines.extend(["", "Previous best attempt:", best_context])

    return "\n".join(lines)


def _examples_from_jsonl(path: Path) -> Iterable[GSM8KExample]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            row = json.loads(line)
            question = row["question"]
            answer = row["answer"]
            example_id = str(row.get("example_id", row.get("id", i)))
            gold = parse_final_answer(answer)
            if gold is None:
                raise ValueError(f"Could not parse gold answer for example {example_id}")
            yield GSM8KExample(
                example_id=example_id,
                question=question,
                answer=answer,
                gold_answer=gold,
            )


def _examples_from_hf(split: str) -> Iterable[GSM8KExample]:
    from datasets import load_dataset  # Imported lazily for offline tests.

    dataset = load_dataset("gsm8k", "main", split=split)
    for i, row in enumerate(dataset):
        gold = parse_final_answer(row["answer"])
        if gold is None:
            raise ValueError(f"Could not parse gold answer for row {i}")
        yield GSM8KExample(
            example_id=str(i),
            question=row["question"],
            answer=row["answer"],
            gold_answer=gold,
        )


def load_gsm8k_examples(
    split: str,
    max_examples: Optional[int] = None,
    local_cache_path: Optional[str] = None,
) -> list[GSM8KExample]:
    if local_cache_path:
        source = list(_examples_from_jsonl(Path(local_cache_path)))
    else:
        source = list(_examples_from_hf(split))

    if max_examples is not None:
        return source[:max_examples]
    return source
