from dataclasses import dataclass

Instructions = str | dict | list


# The checks below return an error message (or None) instead of raising, so the
# error is raised from the question class and the traceback points at the
# caller's own line.

def _instructions_problem(question: str, instructions) -> tuple[type, str] | None:
    if not isinstance(instructions, (str, dict, list)):
        return TypeError, (
            f"{question} instructions must be text, a dict or a list, "
            f"not {type(instructions).__name__}."
        )
    if not (instructions.strip() if isinstance(instructions, str) else instructions):
        return ValueError, (
            f"{question} instructions are empty. Write the question, "
            f'like instructions="Which team should handle this?".'
        )
    return None


def _blank(description) -> bool:
    # None means "no description"; an empty or blank text is almost always a slip.
    return isinstance(description, str) and not description.strip()


class Choice:
    def __init__(
        self,
        *,
        instructions: Instructions,
        criteria: dict[str, str | None] | None = None,
        **options: str | None,
    ):
        if criteria is not None and options:
            raise TypeError("Choice takes either criteria={...} or named options, not both.")
        if problem := _instructions_problem("Choice", instructions):
            raise problem[0](problem[1])
        self.instructions = instructions
        self.criteria = options if criteria is None else criteria
        if len(self.criteria) < 2:
            raise ValueError(
                f"Choice needs at least two options to pick from; it has {len(self.criteria)}. "
                'Add them like Choice(instructions="...", billing=None, technical=None).'
            )
        for name, description in self.criteria.items():
            if _blank(description):
                raise ValueError(
                    f'Choice option "{name}" has an empty description. '
                    f'Write what it covers, like {name}="...", '
                    f"or use {name}=None if it needs no description."
                )

    def to_dict(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True, kw_only=True)
class Score:
    instructions: Instructions
    levels: list[str]

    def __post_init__(self):
        if problem := _instructions_problem("Score", self.instructions):
            raise problem[0](problem[1])
        if not 2 <= len(self.levels) <= 10:
            raise ValueError(
                f"Score needs 2 to 10 levels, lowest first; it has {len(self.levels)}. "
                "For more options without an order, use Choice."
            )
        seen = {}
        for index, level in enumerate(self.levels):
            if isinstance(level, str) and level in seen:
                raise ValueError(
                    f'Score levels {seen[level]} and {index} are both "{level}". '
                    "Give each level its own text, or None."
                )
            if isinstance(level, str):
                seen[level] = index
        for index, level in enumerate(self.levels):
            if _blank(level):
                raise ValueError(
                    f"Score level {index} (counting from 0) is empty. "
                    f'Write what it means, like "Today", or use None to leave it undescribed.'
                )

    def to_dict(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": self.levels}


@dataclass(frozen=True, kw_only=True)
class Noul:
    instructions: Instructions
    true: str | None = None
    false: str | None = None

    def __post_init__(self):
        if problem := _instructions_problem("Noul", self.instructions):
            raise problem[0](problem[1])
        for name in ("true", "false"):
            if _blank(getattr(self, name)):
                raise ValueError(
                    f'Noul {name}="" is empty. Describe what counts as {"yes" if name == "true" else "no"}, '
                    f"or leave {name} out (the same as {name}=None)."
                )

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
