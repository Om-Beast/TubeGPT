import unittest

from rag_core import (
    build_timestamped_chunks,
    extract_video_id,
    format_timestamp,
    parse_pasted_transcript,
)


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


class PastedTranscriptTests(unittest.TestCase):
    def test_parses_inline_timestamps_and_computes_durations(self):
        snippets, has_timestamps = parse_pasted_transcript(
            "0:00 Intro to RAG\n0:12 Embeddings map text to vectors\n1:02:03 Longer explanation"
        )
        self.assertTrue(has_timestamps)
        self.assertEqual(len(snippets), 3)
        self.assertEqual(snippets[0]["start"], 0)
        self.assertEqual(snippets[0]["duration"], 12)
        self.assertEqual(snippets[1]["start"], 12)
        self.assertEqual(snippets[1]["duration"], 61 * 60 + 51)
        self.assertEqual(snippets[2]["start"], 3723)

    def test_parses_standalone_timestamps_followed_by_caption_lines(self):
        snippets, has_timestamps = parse_pasted_transcript(
            "0:00\nHello there\n0:08\nWelcome back"
        )
        self.assertTrue(has_timestamps)
        self.assertEqual([item["text"] for item in snippets], ["Hello there", "Welcome back"])
        self.assertEqual([item["start"] for item in snippets], [0, 8])
        self.assertEqual(snippets[0]["duration"], 8)

    def test_unstamped_text_is_returned_without_fake_timestamps(self):
        snippets, has_timestamps = parse_pasted_transcript("First paragraph\nSecond paragraph")
        self.assertFalse(has_timestamps)
        self.assertEqual(len(snippets), 1)
        self.assertEqual(snippets[0]["text"], "First paragraph Second paragraph")
        self.assertEqual(snippets[0]["start"], 0)

    def test_empty_transcript_is_rejected_as_empty(self):
        snippets, has_timestamps = parse_pasted_transcript("  \n \n")
        self.assertFalse(has_timestamps)
        self.assertEqual(snippets, [])


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
