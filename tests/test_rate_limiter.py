import unittest

from rate_limiter import SlidingWindowRateLimiter


class SlidingWindowRateLimiterTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.limiter = SlidingWindowRateLimiter(
            window_seconds=60,
            max_keys=3,
            clock=lambda: self.now[0],
        )

    def test_allows_requests_up_to_limit_and_returns_retry_after(self):
        self.assertIsNone(self.limiter.check("client-a:/api/chat", 2))
        self.assertIsNone(self.limiter.check("client-a:/api/chat", 2))
        self.assertEqual(self.limiter.check("client-a:/api/chat", 2), 60)

    def test_window_expiry_releases_capacity(self):
        self.limiter.check("client-a", 1)
        self.assertEqual(self.limiter.check("client-a", 1), 60)
        self.now[0] += 60
        self.assertIsNone(self.limiter.check("client-a", 1))

    def test_clients_and_routes_have_independent_budgets(self):
        self.limiter.check("client-a:/api/chat", 1)
        self.assertEqual(self.limiter.check("client-a:/api/chat", 1), 60)
        self.assertIsNone(self.limiter.check("client-b:/api/chat", 1))
        self.assertIsNone(self.limiter.check("client-a:/api/load-video", 1))

    def test_expired_entries_are_removed(self):
        self.limiter.check("client-a", 1)
        self.now[0] += 60.1
        self.assertIsNone(self.limiter.check("client-a", 1))

    def test_key_store_is_bounded(self):
        for key in ("a", "b", "c", "d", "e"):
            self.limiter.check(key, 2)
        self.assertEqual(self.limiter.tracked_keys, 3)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(window_seconds=0)
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(max_keys=0)
        with self.assertRaises(ValueError):
            self.limiter.check("client-a", 0)


if __name__ == "__main__":
    unittest.main()
