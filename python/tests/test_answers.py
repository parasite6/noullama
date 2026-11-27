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


class AllValuesTests(unittest.TestCase):
    def test_returns_everything_with_time(self):
        raw = {**NATIVE, "extra_top": {"x": 1}}
        vals = Decision.from_raw(raw, nl_time=0.25).nl_all_vals()
        self.assertEqual(vals["model"], "d1-2B-Q8_0")
        self.assertEqual(vals["usage"], {"input_tokens": 179, "output_tokens": 0})
        self.assertEqual(vals["nl_time"], 0.25)
        self.assertEqual(vals["extra_top"], {"x": 1})
        self.assertEqual(list(vals["answers"]), ["route", "urgency", "injection", "sensitive"])

        urgency = vals["answers"]["urgency"]
        self.assertEqual(urgency["type"], "score")
        self.assertEqual(urgency["value"], 2.0)
        self.assertEqual(urgency["score"], 2.0)
        self.assertEqual(urgency["legend"]["2"], "Today")
        self.assertEqual(urgency["probabilities"]["3"], 0.19373963911320685)
        self.assertEqual(urgency["confidence"], 0.7987921977234174)

        injection = vals["answers"]["injection"]
        self.assertEqual(injection["value"], 0.9939796403887508)
        self.assertEqual(injection["noul"], 0.9939796403887508)
        self.assertAlmostEqual(injection["probabilities"]["false"], 1 - 0.9939796403887508)
        self.assertEqual(injection["confidence"], 0.9939796403887508)

    def test_missing_fields_are_none_and_time_none_without_timing(self):
        vals = Decision.from_raw({"answers": {"a": {"type": "choice", "choice": "x"}}}).nl_all_vals()
        self.assertEqual((vals["model"], vals["usage"], vals["nl_time"]), (None, None, None))
        self.assertEqual(vals["answers"]["a"], {
            "type": "choice", "choice": "x", "value": "x", "probabilities": None, "confidence": None,
            "label": "x",
        })

    def test_changing_the_result_leaves_the_decision_alone(self):
        d = Decision.from_raw(NATIVE, nl_time=0.1)
        vals = d.nl_all_vals()
        vals["answers"]["urgency"]["legend"]["2"] = "changed"
        vals["answers"]["route"]["probabilities"]["billing"] = 0
        vals["usage"]["input_tokens"] = 0
        self.assertEqual(d["urgency"].raw["legend"]["2"], "Today")
        self.assertEqual(d["route"].probabilities["billing"], 0.9939483837002184)
        self.assertEqual(d.usage["input_tokens"], 179)
        self.assertEqual(NATIVE["usage"]["input_tokens"], 179)


LEVELS = ["Can wait", "Keep watching", "Notify seniors", "Deploy emergency air force unit"]


class LabelTests(unittest.TestCase):
    def sent(self, levels=LEVELS):
        return {"threat": {"type": "score", "instructions": "How bad?", "criteria": levels}}

    def test_score_label_is_the_level_text(self):
        raw = {"answers": {"threat": {"type": "score", "score": 2.0,
               "probabilities": {"0": 0.1, "1": 0.1, "2": 0.7, "3": 0.1}, "confidence": 0.7}}}
        self.assertEqual(Decision.from_raw(raw, questions=self.sent())["threat"].label, "Notify seniors")

    def test_score_label_uses_most_likely_level_not_a_weighted_average(self):
        # Winnow sends a weighted average score; the label is the most likely level.
        raw = {"answers": {"threat": {"type": "score", "score": 1.6,
               "probabilities": {"0": 0.3, "1": 0.0, "2": 0.1, "3": 0.6}, "confidence": 0.4}}}
        self.assertEqual(Decision.from_raw(raw, questions=self.sent())["threat"].label,
                         "Deploy emergency air force unit")

    def test_score_label_without_probabilities_rounds_the_score(self):
        raw = {"answers": {"threat": {"type": "score", "score": 0.6}}}
        self.assertEqual(Decision.from_raw(raw, questions=self.sent())["threat"].label, "Keep watching")

    def test_score_label_falls_back_to_the_reply_legend(self):
        self.assertEqual(Decision.from_raw(NATIVE)["urgency"].label, "Today")

    def test_undescribed_level_and_out_of_range_give_none(self):
        raw = {"answers": {"threat": {"type": "score", "score": 1.0}}}
        self.assertIsNone(Decision.from_raw(raw, questions=self.sent(["a", None]))["threat"].label)
        raw = {"answers": {"threat": {"type": "score", "score": 7.0}}}
        self.assertIsNone(Decision.from_raw(raw, questions=self.sent())["threat"].label)

    def test_choice_label_is_the_option_and_noul_has_none(self):
        d = Decision.from_raw(NATIVE)
        self.assertEqual(d["route"].label, "billing")
        self.assertIsNone(d["injection"].label)

    def test_label_is_in_nl_all_vals(self):
        raw = {"answers": {"threat": {"type": "score", "score": 3.0}}}
        vals = Decision.from_raw(raw, questions=self.sent()).nl_all_vals()
        self.assertEqual(vals["answers"]["threat"]["label"], "Deploy emergency air force unit")
