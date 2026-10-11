import unittest

from retrieval_metrics import aggregate_retrieval_metrics, retrieval_metrics


class RetrievalMetricsTests(unittest.TestCase):
    def test_reciprocal_rank_and_gold_recall(self):
        result = retrieval_metrics([8, 7, 5, 2], [5, 8], k=4)
        self.assertEqual(result["hit_at_k"], 1)
        self.assertEqual(result["gold_recall_at_k"], 1.0)
        self.assertEqual(result["first_relevant_rank"], 1)
        self.assertEqual(result["reciprocal_rank"], 1.0)

    def test_no_hit_returns_zero(self):
        result = retrieval_metrics([1, 2], [9], k=2)
        self.assertEqual(result["hit_at_k"], 0)
        self.assertEqual(result["gold_recall_at_k"], 0.0)
        self.assertEqual(result["reciprocal_rank"], 0.0)

    def test_aggregate_reports_query_count(self):
        result = aggregate_retrieval_metrics([
            {"retrieved_chunk_ids": [1, 2], "gold_chunk_ids": [2]},
            {"retrieved_chunk_ids": [4], "gold_chunk_ids": [4]},
        ], k=2)
        self.assertEqual(result["queries_evaluated"], 2)
        self.assertEqual(result["hit_rate_at_k"], 1.0)
        self.assertAlmostEqual(result["mean_reciprocal_rank"], 0.75)

    def test_invalid_cases_fail_fast(self):
        with self.assertRaises(ValueError):
            retrieval_metrics([1], [], k=1)
        with self.assertRaises(ValueError):
            aggregate_retrieval_metrics([], k=4)


if __name__ == "__main__":
    unittest.main()
