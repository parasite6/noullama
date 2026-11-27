from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class Answer:
    type: str
    value: Any
    probabilities: dict[str, float] | None
    confidence: float | None
    raw: dict

    @classmethod
    def from_raw(cls, raw: dict) -> "Answer":
        kind = raw["type"]
        value = raw.get(kind)
        probabilities = raw.get("probabilities")
        confidence = raw.get("confidence")
        if kind == "noul":
            if "probabilities" not in raw:
                probabilities = {"false": 1 - value, "true": value}
            if "confidence" not in raw:
                confidence = max(value, 1 - value)
        return cls(kind, value, probabilities, confidence, raw)


@dataclass
class Decision:
    answers: dict[str, Answer]
    model: str | None
    usage: dict | None
    raw: dict
    nl_time: float | None = None
    """Seconds from sending the request to reading the full response."""

    @classmethod
    def from_raw(cls, raw: dict, *, nl_time: float | None = None) -> "Decision":
        answers = {name: Answer.from_raw(answer) for name, answer in raw["answers"].items()}
        return cls(answers, raw.get("model"), raw.get("usage"), raw, nl_time)

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.answers

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)
