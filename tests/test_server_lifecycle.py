"""Strict startup and resource cleanup without loading any learned models."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import server


class FakeEngine:
    # Purpose: Configure startup readiness/failure cases without loading inference models.
    # Input: strict: require multimodal readiness; ready: reported readiness; load_error: optional
    #          startup exception.
    # Output: None; stores settings/errors/readiness and initializes the close flag.
    def __init__(self, *, strict=True, ready=True, load_error=None):
        self.settings = SimpleNamespace(require_multimodal=strict)
        self.errors = {"text_emotion": "missing checkpoint"} if not ready else {}
        self.ready = ready
        self.load_error = load_error
        self.closed = False

    # Purpose: Simulate successful or failing startup.
    # Input: None explicitly; uses the configured load_error.
    # Output: None, or raises the configured startup exception.
    def load(self):
        if self.load_error:
            raise self.load_error

    # Purpose: Expose the synthetic Engine readiness flag.
    # Input: None explicitly; uses self.ready.
    # Output: Dict containing multimodal_ready.
    def info(self):
        return {"multimodal_ready": self.ready}

    # Purpose: Record that server shutdown released the Engine.
    # Input: None.
    # Output: None; sets closed to True.
    def close(self):
        self.closed = True


class ServerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    # Purpose: Verify strict mode rejects missing models and closes workers.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_strict_mode_rejects_missing_models_and_closes_workers(self):
        engine = FakeEngine(ready=False)
        with patch.object(server, "engine", engine):
            with self.assertRaisesRegex(RuntimeError, "Required multimodal mode"):
                async with server.lifespan(server.app):
                    self.fail("Strict startup must fail before serving")
        self.assertTrue(engine.closed)

    # Purpose: Verify ready and explicit fallback modes start and close.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_ready_and_explicit_fallback_modes_start_and_close(self):
        for strict, ready in [(True, True), (False, False)]:
            engine = FakeEngine(strict=strict, ready=ready)
            with self.subTest(strict=strict, ready=ready), patch.object(server, "engine", engine):
                async with server.lifespan(server.app):
                    self.assertFalse(engine.closed)
            self.assertTrue(engine.closed)

    # Purpose: Verify load failure closes workers.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_load_failure_closes_workers(self):
        engine = FakeEngine(load_error=RuntimeError("load failed"))
        with patch.object(server, "engine", engine):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                async with server.lifespan(server.app):
                    self.fail("Model load failure must propagate")
        self.assertTrue(engine.closed)


class BrowserUrl(unittest.TestCase):
    # Purpose: Verify the startup message shows an address a browser can open.
    # Input: None explicitly.
    # Output: None; raises an assertion if the URL format regresses.
    def test_browser_url(self):
        self.assertEqual(server.browser_url("127.0.0.1", 8000), "http://127.0.0.1:8000")
        self.assertEqual(server.browser_url("0.0.0.0", 8080), "http://127.0.0.1:8080")
        self.assertEqual(server.browser_url("::1", 9000), "http://[::1]:9000")


if __name__ == "__main__":
    unittest.main()
