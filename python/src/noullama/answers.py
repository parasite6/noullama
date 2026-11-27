import copy
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
    label: str | None = None
    """Choice: the picked option. Score: the text of the most likely level. Noul: None."""

    @classmethod
    def from_raw(cls, raw: dict, question: dict | None = None) -> "Answer":
        kind = raw["type"]
        value = raw.get(kind)
        probabilities = raw.get("probabilities")
        confidence = raw.get("confidence")
        if kind == "noul":
            if "probabilities" not in raw:
                probabilities = {"false": 1 - value, "true": value}
            if "confidence" not in raw:
                confidence = max(value, 1 - value)
        label = None
        if kind == "choice" and isinstance(value, str):
            label = value
        elif kind == "score":
            label = _score_label(value, probabilities, raw, question)
        return cls(kind, value, probabilities, confidence, raw, label)


def _score_label(value, probabilities, raw: dict, question: dict | None) -> str | None:
    # The most likely level: native models send it as the score, while Winnow sends a
    # weighted average, so prefer the level with the highest probability.
    index = None
    if isinstance(probabilities, dict) and probabilities:
        best = max(probabilities, key=probabilities.get)
        if best.isdigit():
            index = int(best)
    if index is None and isinstance(value, (int, float)) and not isinstance(value, bool):
        index = int(value + 0.5)
    if index is None or index < 0:
        return None
    levels = question.get("criteria") if isinstance(question, dict) else None
    if isinstance(levels, list):
        level = levels[index] if index < len(levels) else None
    else:
        legend = raw.get("legend")
        level = legend.get(str(index)) if isinstance(legend, dict) else None
    return level if isinstance(level, str) else None


@dataclass
class Decision:
    answers: dict[str, Answer]
    model: str | None
    usage: dict | None
    raw: dict
    nl_time: float | None = None
    """Seconds from sending the request to reading the full response."""

    @classmethod
    def from_raw(
        cls, raw: dict, *, nl_time: float | None = None, questions: dict[str, dict] | None = None
    ) -> "Decision":
        questions = questions or {}
        answers = {
            name: Answer.from_raw(answer, questions.get(name)) for name, answer in raw["answers"].items()
        }
        return cls(answers, raw.get("model"), raw.get("usage"), raw, nl_time)

    def nl_all_vals(self) -> dict:
        """Everything in the decision as plain data: the reply's own fields, nl_time,
        and every answer's raw fields plus type, value, probabilities, confidence and label."""
        values = {key: copy.deepcopy(item) for key, item in self.raw.items() if key != "answers"}
        values.update(
            model=self.model,
            usage=copy.deepcopy(self.usage),
            nl_time=self.nl_time,
            answers={
                name: {
                    **copy.deepcopy(answer.raw),
                    "type": answer.type,
                    "value": copy.deepcopy(answer.value),
                    "probabilities": copy.deepcopy(answer.probabilities),
                    "confidence": answer.confidence,
                    "label": answer.label,
                }
                for name, answer in self.answers.items()
            },
        )
        return values

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.answers

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)
