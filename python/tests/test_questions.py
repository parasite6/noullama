import unittest

from noullama.questions import Choice, Noul, Score, question_dict


class QuestionTests(unittest.TestCase):
    def test_choice_keeps_order_and_none(self):
        self.assertEqual(Choice(instructions="Team?", billing="Refunds", technical=None).to_dict(),
            {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Refunds", "technical": None}})
        self.assertEqual(list(Choice(instructions="x", criteria={"z": None, "a": None}).to_dict()["criteria"]), ["z", "a"])

    def test_choice_rejects_both_forms(self):
        with self.assertRaises(TypeError):
            Choice(instructions="x", criteria={"a": None}, b=None)

    def test_score_sends_levels_as_criteria(self):
        self.assertEqual(Score(instructions="Urgent?", levels=["Can wait", "Now"]).to_dict(),
            {"type": "score", "instructions": "Urgent?", "criteria": ["Can wait", "Now"]})

    def test_noul_criteria_only_when_given(self):
        self.assertEqual(Noul(instructions="Refund?").to_dict(), {"type": "noul", "instructions": "Refund?"})
        self.assertEqual(Noul(instructions="Refund?", true="asks for money back").to_dict()["criteria"], {"true": "asks for money back"})

    def test_question_dict(self):
        raw = {"type": "future", "instructions": "x"}
        self.assertIs(question_dict(raw), raw)
        with self.assertRaises(TypeError):
            question_dict("not a question")

    def test_instructions_cannot_be_empty(self):
        for make in (lambda i: Choice(instructions=i, a=None, b=None),
                     lambda i: Score(instructions=i, levels=["a", "b"]),
                     lambda i: Noul(instructions=i)):
            for empty in ("", "   ", {}, []):
                with self.assertRaises(ValueError):
                    make(empty)
            with self.assertRaises(TypeError):
                make(None)
            make({"question": "x"})  # dicts and lists with content are fine

    def test_descriptions_cannot_be_blank(self):
        for make in (lambda d: Choice(instructions="x", billing=d, technical=None),
                     lambda d: Choice(instructions="x", criteria={"billing": d, "technical": None}),
                     lambda d: Score(instructions="x", levels=[d, "Now"]),
                     lambda d: Noul(instructions="x", true=d)):
            for blank in ("", "   ", "\n"):
                with self.assertRaises(ValueError) as cm:
                    make(blank)
                self.assertIn("None", str(cm.exception))
        # None still means "no description"
        self.assertIsNone(Choice(instructions="x", billing=None, technical=None).to_dict()["criteria"]["billing"])
        self.assertNotIn("criteria", Noul(instructions="x", true=None).to_dict())

    def test_blank_description_names_the_option(self):
        with self.assertRaises(ValueError) as cm:
            Choice(instructions="x", billing="ok", technical="")
        self.assertIn("technical", str(cm.exception))
