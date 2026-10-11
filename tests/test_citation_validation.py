import unittest

from citation_validation import validate_citations


class CitationValidationTests(unittest.TestCase):
    def test_accepts_only_existing_source_ids(self):
        result = validate_citations("Use this approach [S1] and verify [S2].", ["S1", "S2"])
        self.assertTrue(result["valid"])
        self.assertTrue(result["has_citations"])
        self.assertEqual(result["used_source_ids"], ["S1", "S2"])
        self.assertEqual(result["invalid_source_ids"], [])

    def test_flags_unknown_sources(self):
        result = validate_citations("This is supported [S1], but also [S9].", ["S1", "S2"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["used_source_ids"], ["S1"])
        self.assertEqual(result["invalid_source_ids"], ["S9"])

    def test_missing_citations_are_not_marked_valid(self):
        result = validate_citations("Here is a response without labels.", ["S1"])
        self.assertFalse(result["has_citations"])
        self.assertFalse(result["valid"])

    def test_deduplicates_labels_but_flags_case_mismatch(self):
        result = validate_citations("[S1] [S1] [s1]", ["S1"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["used_source_ids"], ["S1"])
        self.assertEqual(result["invalid_source_ids"], ["s1"])


if __name__ == "__main__":
    unittest.main()
