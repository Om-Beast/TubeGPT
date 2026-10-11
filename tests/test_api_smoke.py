import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app as application
from rate_limiter import SlidingWindowRateLimiter


VIDEO_ID = "Gfr50f6ZBvo"


class ApiSmokeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
        self.environment.start()
        self.client = application.app.test_client()
        self.rate_limiter_patch = patch.object(application, "_RATE_LIMITER", SlidingWindowRateLimiter())
        self.rate_limiter_patch.start()
        with application._CACHE_LOCK:
            application._INDEX_CACHE.clear()

    def tearDown(self):
        with application._CACHE_LOCK:
            application._INDEX_CACHE.clear()
        self.rate_limiter_patch.stop()
        self.environment.stop()

    def test_health_endpoint_and_security_headers(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")

    def test_invalid_video_url_is_rejected(self):
        response = self.client.post("/api/load-video", json={"video_url": "https://example.com/not-youtube"})
        self.assertEqual(response.status_code, 400)

    def test_pasted_transcript_route_parses_and_indexes_through_shared_path(self):
        entry = {
            "language": "pasted transcript (timestamps preserved)",
            "word_count": 5,
            "chunk_count": 1,
            "timestamp_available": True,
        }
        with patch.object(application, "_build_index", return_value=(entry, False)) as build_index:
            response = self.client.post("/api/load-transcript", json={
                "video_url": "https://www.youtube.com/watch?v=" + VIDEO_ID,
                "transcript_text": "0:00 Intro to RAG\n0:10 RAG retrieves evidence",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["input_mode"], "pasted")
        self.assertTrue(response.get_json()["timestamp_available"])
        build_index.assert_called_once()

    def test_chat_repairs_unknown_citation_and_returns_audit(self):
        source = {
            "id": "S1",
            "chunk_index": 0,
            "text": "RAG retrieves relevant passages before generation.",
            "start_seconds": 0,
            "end_seconds": 10,
            "timestamp_available": True,
            "start_timestamp": "00:00",
            "end_timestamp": "00:10",
            "youtube_url": "https://www.youtube.com/watch?v=" + VIDEO_ID + "&t=0s",
        }
        with application._CACHE_LOCK:
            application._INDEX_CACHE[VIDEO_ID] = {"store": object(), "documents": []}

        model = SimpleNamespace(invoke=Mock(side_effect=[
            SimpleNamespace(content="RAG retrieves the whole internet [S9]."),
            SimpleNamespace(content="RAG retrieves relevant passages first [S1]."),
        ]))
        with patch.object(application, "_retrieve_sources", return_value=([source], "[S1] RAG retrieves relevant passages.")), \
             patch.object(application, "ChatGoogleGenerativeAI", return_value=model):
            response = self.client.post("/api/chat", json={
                "video_id": VIDEO_ID,
                "question": "How does RAG work?",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["citation_audit"]["valid"])
        self.assertEqual(payload["citation_audit"]["attempts"], 2)
        self.assertEqual(model.invoke.call_count, 2)
        self.assertEqual(payload["sources"][0]["id"], "S1")

    def test_chat_without_index_fails_with_conflict(self):
        response = self.client.post("/api/chat", json={
            "video_id": VIDEO_ID,
            "question": "What happened?",
        })
        self.assertEqual(response.status_code, 409)

    def test_rate_limit_returns_retry_after(self):
        for _ in range(4):
            response = self.client.post(
                "/api/load-video",
                json={"video_url": "bad"},
                environ_base={"REMOTE_ADDR": "198.51.100.44"},
            )
            self.assertEqual(response.status_code, 400)
        response = self.client.post(
            "/api/load-video",
            json={"video_url": "bad"},
            environ_base={"REMOTE_ADDR": "198.51.100.44"},
        )
        self.assertEqual(response.status_code, 429)
        self.assertTrue(response.headers.get("Retry-After"))


if __name__ == "__main__":
    unittest.main()
