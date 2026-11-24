import unittest

from fetchers.generic import _extract_fields


class GenericFieldExtractionTests(unittest.TestCase):
    def test_progress_field_extracts_used_and_total_values(self):
        payload = {
            "this_month_usage": 1,
            "searches_per_month": 250,
        }
        fields = [{
            "label": "Monthly searches",
            "path": "this_month_usage",
            "total_path": "searches_per_month",
            "type": "progress",
        }]

        self.assertEqual(_extract_fields(payload, fields), [{
            "label": "Monthly searches",
            "path": "this_month_usage",
            "type": "progress",
            "symbol": "", "history": False,
            "value": 1,
            "total": 250,
        }])

    def test_all_progress_variants_extract_total_values(self):
        payload = {"value": 75, "limit": 100}

        for field_type in (
            "progress",
            "progress_remaining",
            "progress_bar",
            "progress_bar_remaining",
        ):
            with self.subTest(field_type=field_type):
                fields = [{
                    "label": "Quota",
                    "path": "value",
                    "total_path": "limit",
                    "type": field_type,
                }]

                self.assertEqual(_extract_fields(payload, fields), [{
                    "label": "Quota",
                    "path": "value",
                    "type": field_type,
                    "symbol": "", "history": False,
                    "value": 75,
                    "total": 100,
                }])

    def test_existing_field_types_are_unchanged(self):
        fields = [{"label": "Plan", "path": "plan.name", "type": "text"}]

        self.assertEqual(_extract_fields({"plan": {"name": "Free"}}, fields), [{
            "label": "Plan",
            "path": "plan.name",
            "type": "text",
            "symbol": "", "history": False,
            "value": "Free",
        }])


class GenericDivisorTests(unittest.TestCase):
    def test_new_api_quota_is_scaled_once(self):
        fields = [{"label": "余额", "path": "quota", "type": "balance", "divisor": 500000}]
        self.assertAlmostEqual(_extract_fields({"quota": 23494295}, fields)[0]["value"], 46.98859)

    def test_default_and_request_count_are_unchanged(self):
        for kind in ("number", "balance", "currency"):
            field = {"label": "值", "path": "value", "type": kind}
            self.assertEqual(_extract_fields({"value": 3771}, [field])[0]["value"], 3771)

    def test_numeric_strings_and_missing_values(self):
        field = {"label": "值", "path": "value", "type": "number", "divisor": 500000}
        self.assertEqual(_extract_fields({"value": "1000000"}, [field])[0]["value"], 2)
        for value in (None, "unknown", True):
            self.assertEqual(_extract_fields({"value": value}, [field])[0]["value"], value)

    def test_invalid_divisors_are_rejected(self):
        for divisor in (0, -1, float("nan"), float("inf"), "500000", True):
            with self.subTest(divisor=divisor), self.assertRaises(ValueError):
                _extract_fields({"value": 1}, [{"label": "值", "path": "value", "divisor": divisor}])

    def test_percentages_are_not_scaled(self):
        field = {"label": "用量", "path": "value", "type": "percentage_fraction", "divisor": 500000}
        self.assertEqual(_extract_fields({"value": 0.69}, [field])[0]["value"], 0.69)


if __name__ == "__main__":
    unittest.main()
