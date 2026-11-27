import unittest

from noullama.answers import Answer, Decision


NATIVE = {"model": "d1-2B-Q8_0", "answers": {
    "route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9939483837002184, "technical": 0.0017474845175467018, "shipping": 0.0007090222365840068, "general": 0.003595109545650952}, "confidence": 0.9939483837002184},
    "urgency": {"type": "score", "score": 2.0, "legend": {"0": "Can wait", "1": "This week", "2": "Today", "3": "Immediately; blocked or significant ongoing impact"}, "probabilities": {"0": 0.0032892468268656363, "1": 0.004178916336510071, "2": 0.7987921977234174, "3": 0.19373963911320685}, "confidence": 0.7987921977234174},
    "injection": {"type": "noul", "noul": 0.9939796403887508},
    "sensitive": {"type": "noul", "noul": 0.012670335639728072}},
  "usage": {"input_tokens": 179, "output_tokens": 0}}


class AnswerTests(unittest.TestCase):
    def test_choice_and_score_pass_through(self):
        d = Decision.from_raw(NATIVE)
        self.assertEqual(d["route"].value, "billing")
        self.assertEqual(d["route"].confidence, 0.9939483837002184)
        self.assertEqual(d["urgency"].value, 2.0)
        self.assertEqual(d["urgency"].raw["legend"]["2"], "Today")
        self.assertEqual(d["urgency"].probabilities["3"], 0.19373963911320685)

    def test_noul_fill_in(self):
        d = Decision.from_raw(NATIVE)
        self.assertAlmostEqual(d["injection"].probabilities["false"], 1 - 0.9939796403887508)
        self.assertEqual(d["injection"].confidence, 0.9939796403887508)
        self.assertAlmostEqual(d["sensitive"].confidence, 1 - 0.012670335639728072)

    def test_noul_gateway_values_win(self):
        a = Answer.from_raw({"type": "noul", "noul": 0.7, "probabilities": {"false": 0.25, "true": 0.75}, "confidence": 0.5})
        self.assertEqual((a.probabilities["false"], a.confidence), (0.25, 0.5))

    def test_choice_missing_fields_stay_none(self):
        a = Answer.from_raw({"type": "choice", "choice": "a"})
        self.assertIsNone(a.probabilities)
        self.assertIsNone(a.confidence)

    def test_unknown_type_passes_through(self):
        a = Answer.from_raw({"type": "extract", "extract": "INV-2048", "extra": 1})
        self.assertEqual((a.value, a.raw["extra"]), ("INV-2048", 1))

    def test_decision_container(self):
        d = Decision.from_raw(NATIVE)
        self.assertEqual((d.model, d.usage["input_tokens"], len(d), "route" in d), ("d1-2B-Q8_0", 179, 4, True))
        self.assertEqual(list(d), ["route", "urgency", "injection", "sensitive"])
