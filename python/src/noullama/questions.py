from dataclasses import dataclass

Instructions = str | dict | list


def _check_instructions(instructions) -> None:
    if not isinstance(instructions, (str, dict, list)):
        raise TypeError("instructions must be text, a dict or a list.")
    if not (instructions.strip() if isinstance(instructions, str) else instructions):
        raise ValueError("instructions can't be empty.")


def _check_description(name: str, description) -> None:
    # None means "no description"; an empty or blank text is almost always a slip.
    if isinstance(description, str) and not description.strip():
        raise ValueError(f"{name} has an empty description; write one, or use None for none.")


class Choice:
    def __init__(
        self,
        *,
        instructions: Instructions,
        criteria: dict[str, str | None] | None = None,
        **options: str | None,
    ):
        if criteria is not None and options:
            raise TypeError("Pass criteria or keyword options, not both.")
        _check_instructions(instructions)
        self.instructions = instructions
        self.criteria = options if criteria is None else criteria
        for name, description in self.criteria.items():
            _check_description(name, description)

    def to_dict(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True, kw_only=True)
class Score:
    instructions: Instructions
    levels: list[str]

    def __post_init__(self):
        _check_instructions(self.instructions)
        for index, level in enumerate(self.levels):
            _check_description(f"Level {index}", level)

    def to_dict(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": self.levels}


@dataclass(frozen=True, kw_only=True)
class Noul:
    instructions: Instructions
    true: str | None = None
    false: str | None = None

    def __post_init__(self):
        _check_instructions(self.instructions)
        _check_description("true", self.true)
        _check_description("false", self.false)

    def to_dict(self) -> dict:
        result = {"type": "noul", "instructions": self.instructions}
        criteria = {key: value for key, value in (("true", self.true), ("false", self.false))
                    if value is not None}
        if criteria:
            result["criteria"] = criteria
        return result


def question_dict(q: Choice | Score | Noul | dict) -> dict:
    if isinstance(q, dict):
        return q
    if isinstance(q, (Choice, Score, Noul)):
        return q.to_dict()
    raise TypeError("Expected a Choice, Score, Noul or dict.")
