import unittest

from hybrid_retrieval import bm25_scores, hybrid_rank_documents, reciprocal_rank_fusion, tokenize


def doc(index, text):
    return {"page_content": text, "metadata": {"chunk_index": index}}


class HybridRetrievalTests(unittest.TestCase):
    def test_tokenizer_supports_unicode_and_punctuation(self):
        self.assertEqual(tokenize("RAG, Gemini-3 Hindi!"), ["rag", "gemini", "3", "hindi"])

    def test_bm25_promotes_exact_technical_terms(self):
        corpus = [
            doc(0, "General introduction to video learning"),
            doc(1, "FAISS vector index and BM25 hybrid retrieval"),
            doc(2, "Python backend and Flask API"),
        ]
        scores = bm25_scores("BM25 retrieval FAISS", corpus)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(scores[0], 0.0)

    def test_hybrid_uses_dense_and_lexical_evidence(self):
        corpus = [
            doc(0, "A general explanation"),
            doc(1, "The exact error code is RATE_LIMIT_429"),
            doc(2, "FAISS uses vectors for semantic retrieval"),
        ]
        dense = [corpus[2], corpus[0]]
        ranked = hybrid_rank_documents("RATE_LIMIT_429", dense, corpus, top_k=3)
        self.assertEqual([item["metadata"]["chunk_index"] for item in ranked][0], 1)
        self.assertIn(2, [item["metadata"]["chunk_index"] for item in ranked])

    def test_rrf_deduplicates_chunk_ids(self):
        a, b = doc(1, "same chunk"), doc(2, "other chunk")
        ranked = reciprocal_rank_fusion([[a, b], [a]], top_k=3)
        self.assertEqual([item["metadata"]["chunk_index"] for item in ranked], [1, 2])

    def test_empty_lexical_match_falls_back_to_dense(self):
        corpus = [doc(0, "some useful text"), doc(1, "other text")]
        ranked = hybrid_rank_documents("unmatchedtoken", [corpus[1], corpus[0]], corpus, top_k=1)
        self.assertEqual(ranked[0]["metadata"]["chunk_index"], 1)

    def test_bad_parameters_raise(self):
        with self.assertRaises(ValueError):
            bm25_scores("x", [], k1=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[doc(1, "x")]], top_k=0)


if __name__ == "__main__":
    unittest.main()
