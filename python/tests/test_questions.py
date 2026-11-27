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
