import unittest

from rag_core import build_timestamped_chunks, extract_video_id, format_timestamp


class VideoIdTests(unittest.TestCase):
    def test_accepts_common_youtube_urls(self):
        video_id = "Gfr50f6ZBvo"
        examples = [
            video_id,
            "https://www.youtube.com/watch?v=" + video_id + "&list=playlist",
            "https://youtu.be/" + video_id + "?t=42",
            "https://youtube.com/shorts/" + video_id,
            "https://www.youtube.com/embed/" + video_id,
            "youtube.com/live/" + video_id,
        ]
        for value in examples:
            with self.subTest(value=value):
                self.assertEqual(extract_video_id(value), video_id)

    def test_rejects_invalid_ids_and_untrusted_hosts(self):
        self.assertIsNone(extract_video_id("https://youtube.com.evil.example/watch?v=Gfr50f6ZBvo"))
        self.assertIsNone(extract_video_id("https://example.com/watch?v=Gfr50f6ZBvo"))
        self.assertIsNone(extract_video_id("short"))
        self.assertIsNone(extract_video_id("https://youtu.be/not-an-id"))


class TimestampTests(unittest.TestCase):
    def test_formats_seconds(self):
        self.assertEqual(format_timestamp(0), "00:00")
        self.assertEqual(format_timestamp(79.9), "01:19")
        self.assertEqual(format_timestamp(3661), "01:01:01")
        self.assertEqual(format_timestamp(-2), "00:00")
        self.assertEqual(format_timestamp(float("nan")), "00:00")


class ChunkingTests(unittest.TestCase):
    def test_preserves_time_ranges_and_orders_chunks(self):
        snippets = [
            {"text": "alpha " * 12, "start": 0, "duration": 4},
            {"text": "bravo " * 12, "start": 4, "duration": 5},
            {"text": "charlie " * 12, "start": 9, "duration": 6},
            {"text": "delta " * 12, "start": 15, "duration": 3},
        ]
        chunks = build_timestamped_chunks(snippets, max_chars=180, overlap_chars=40)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], list(range(len(chunks))))
        self.assertEqual(chunks[0]["start_seconds"], 0)
        self.assertTrue(all(chunk["end_seconds"] >= chunk["start_seconds"] for chunk in chunks))
        self.assertTrue(all(chunk["text"] for chunk in chunks))

    def test_ignores_empty_snippets_and_splits_long_captions(self):
        chunks = build_timestamped_chunks(
            [{"text": "", "start": 0, "duration": 1},
             {"text": "word " * 80, "start": 10, "duration": 3}],
            max_chars=100,
            overlap_chars=10,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["text"]) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk["start_seconds"] == 10 for chunk in chunks))

    def test_invalid_chunk_configuration_fails_fast(self):
        with self.assertRaises(ValueError):
            build_timestamped_chunks([], max_chars=20)
        with self.assertRaises(ValueError):
            build_timestamped_chunks([], max_chars=100, overlap_chars=100)


if __name__ == "__main__":
    unittest.main()
