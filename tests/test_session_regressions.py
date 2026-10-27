"""Concurrency regressions with injected models; no model downloads or GPU required.

    python -m unittest tests.test_session_regressions
"""
import asyncio
import threading
import unittest
from unittest.mock import patch

import numpy as np

from app import pipeline
from app.config import SAMPLE_RATE, Settings
from app.llm import TemplateResponder
from tests.test_pipeline import EnergyVAD, FakeASR, FakeEmotion, FakeText, FakeVision, fake_prosody, silence, tone


class BlockingEmotion(FakeEmotion):
    # Purpose: Add controllable synchronization to the fake audio classifier.
    # Input: None.
    # Output: None; initializes started/release thread events and inherited call tracking.
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    # Purpose: Pause voice inference until the test releases the worker.
    # Input: wav: waveform fixture.
    # Output: Fake voice reading; raises TimeoutError if the test never releases it.
    def predict(self, wav):
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError("test did not release the emotion worker")
        return super().predict(wav)


class BlockingResponder:
    repo = "fake-local-responder"

    # Purpose: Prepare generation synchronization and per-call delta counters.
    # Input: None.
    # Output: None; initializes started/release events and call counters.
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.tokens_by_call = []

    # Purpose: Simulate a reply worker that can be paused during reset/cancellation.
    # Input: messages: accepted chat fixture.
    # Output: Generator of 'hello ' deltas; records generated counts and times out if the first call
    #          remains blocked.
    def stream(self, messages):
        index = len(self.tokens_by_call)
        self.tokens_by_call.append(0)
        if index == 0:
            self.started.set()
            if not self.release.wait(2):
                raise TimeoutError("test did not release the reply worker")
        for _ in range(20):
            self.tokens_by_call[index] += 1
            yield "hello "


class SessionRegressions(unittest.IsolatedAsyncioTestCase):
    # Purpose: Replace prosody with deterministic measurements and initialize resource tracking.
    # Input: None.
    # Output: Awaited None; starts the patch and prepares session/Engine/blocker lists.
    async def asyncSetUp(self):
        self.prosody_patch = patch("app.pipeline.prosody.analyze", side_effect=fake_prosody)
        self.prosody_patch.start()
        self.engines = []
        self.sessions = []
        self.blockers = []

    # Purpose: Release blocked workers, close sessions/executors, and restore prosody.
    # Input: None explicitly; uses tracked test resources.
    # Output: Awaited None; drains tasks and releases fixture resources.
    async def asyncTearDown(self):
        for blocker in self.blockers:
            blocker.release.set()
        for session in self.sessions:
            session.close()
            await session.drain(2)
        for engine in self.engines:
            for pool in (engine.torch_pool, engine.mlx_pool, engine.vision_pool):
                await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
        self.prosody_patch.stop()

    # Purpose: Create a tracked fake conversation with optional blocking components.
    # Input: emotion/responder: optional component overrides; history_turns: retention setting.
    # Output: Session fixture; records its Engine and session for teardown.
    def session(self, emotion=None, responder=None, history_turns=3):
        settings = Settings(history_turns=history_turns)
        engine = pipeline.Engine(settings, emotion=emotion or FakeEmotion(), asr=FakeASR(),
                                 responder=responder or TemplateResponder(), vad_factory=EnergyVAD,
                                 text=FakeText(), vision=FakeVision())
        self.engines.append(engine)
        session = pipeline.Session(engine)
        self.sessions.append(session)
        return session

    # Purpose: Wait briefly for a blocking worker to begin.
    # Input: blocker: fixture exposing a started threading.Event.
    # Output: Awaited None; raises on a one-second timeout.
    async def wait_started(self, blocker):
        # Purpose: Poll the captured worker's start signal without blocking asyncio.
        # Input: None explicitly; captures blocker.
        # Output: Awaited None once started is set.
        async def wait():
            while not blocker.started.is_set():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait(), 1)

    # Purpose: Drain already queued events for assertions.
    # Input: session: Session fixture.
    # Output: List of queued event dicts; removes them from session.out.
    @staticmethod
    def events(session):
        return [session.out.get_nowait() for _ in range(session.out.qsize())]

    # Purpose: Verify partial keeps the captured audio and face interval.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_partial_keeps_the_captured_audio_and_face_interval(self):
        emotion = BlockingEmotion()
        self.blockers.append(emotion)
        session = self.session(emotion=emotion)
        session.samples_seen = 3 * SAMPLE_RATE
        session.in_speech = True
        old_face = FakeVision().analyze(None)
        session.add_face_result(2.0, old_face)
        session._spawn(session._partial(1, silence(3), 0.0, 3.0,
                                       session._epoch, session._speech_generation))
        await self.wait_started(emotion)
        # Input keeps arriving while the CNN runs. The new face belongs to a later window.
        session.samples_seen = 6 * SAMPLE_RATE
        new_probs = {name: (0.8 if name == "joy" else 0.2 / 6) for name in old_face["probs"]}
        session.add_face_result(5.0, dict(old_face, probs=new_probs))
        emotion.release.set()
        await session.drain(2)
        events = self.events(session)
        self.assertEqual([event["type"] for event in events], ["partial_state"])
        self.assertEqual(events[0]["modalities"]["face"]["emotion"], "anger")
        self.assertEqual(events[0]["modalities"]["face"]["frames"], 1)

    # Purpose: Verify partial finishing after speech end is discarded.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_partial_finishing_after_speech_end_is_discarded(self):
        emotion = BlockingEmotion()
        self.blockers.append(emotion)
        session = self.session(emotion=emotion)
        session.in_speech = True
        session._spawn(session._partial(1, silence(1), 0.0, 1.0,
                                       session._epoch, session._speech_generation))
        await self.wait_started(emotion)
        session.in_speech = False
        emotion.release.set()
        await session.drain(2)
        self.assertEqual(self.events(session), [])

    # Purpose: Verify reset cancels active and queued turns and session can continue.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_reset_cancels_active_and_queued_turns_and_session_can_continue(self):
        responder = BlockingResponder()
        self.blockers.append(responder)
        session = self.session(responder=responder)
        session.turn_id = 2
        session.history.append({"transcript": "old", "emotion": "neutral", "reply": "old"})
        session._spawn(session.process_utterance(1, tone(0.6)))
        await self.wait_started(responder)
        session._spawn(session.process_utterance(2, tone(0.7)))
        await asyncio.sleep(0)  # put the second turn behind the first turn's lock
        session.feed(tone(0.12))
        session.add_face_result(session.now_s, FakeVision().analyze(None))
        clock_before_reset = session.samples_seen
        session.reset()
        self.assertEqual([event["type"] for event in self.events(session)], ["reset"])
        self.assertFalse(session.in_speech)
        self.assertEqual(session.pending.size, 0)
        self.assertEqual(list(session.pre_roll), [])
        self.assertEqual(session.utt, [])
        self.assertEqual(list(session.faces), [])
        self.assertEqual(session.history, [])
        self.assertEqual(session.samples_seen, clock_before_reset)
        self.assertEqual(session.turn_id, 2)
        await asyncio.sleep(0)  # cancellation propagates to the token bridge
        responder.release.set()
        await session.e.run_mlx(lambda: None)  # wait for the interrupted worker to stop
        await session.drain(2)
        self.assertEqual(self.events(session), [])
        self.assertEqual(session.history, [])
        self.assertEqual(len(responder.tokens_by_call), 1)  # queued turn never generated
        self.assertLessEqual(responder.tokens_by_call[0], 1)

        stream = np.concatenate([tone(0.7), silence(0.7)])
        for i in range(0, stream.size, SAMPLE_RATE // 10):
            session.feed(stream[i:i + SAMPLE_RATE // 10])
            await asyncio.sleep(0)
        session.flush()
        await session.drain(2)
        events = self.events(session)
        finals = [event for event in events if event["type"] == "final_state"]
        dones = [event for event in events if event["type"] == "response_done"]
        self.assertEqual([event["utterance_id"] for event in finals], [3])
        self.assertEqual([event["utterance_id"] for event in dones], [3])
        self.assertEqual(len(session.history), 1)

    # Purpose: Verify history retention obeys the configured limit including zero.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_history_retention_obeys_the_configured_limit_including_zero(self):
        for limit in (0, 2):
            with self.subTest(history_turns=limit):
                session = self.session(history_turns=limit)
                for utterance_id in range(1, 4):
                    await session.process_utterance(utterance_id, tone(0.2 * utterance_id))
                self.assertEqual(len(session.history), limit)
                if limit:
                    self.assertEqual(session.history[-1]["transcript"], "utterance of 0.6 seconds")

    # Purpose: Verify readiness exposes degraded modalities and template responses.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    async def test_readiness_exposes_degraded_modalities_and_template_responses(self):
        session = self.session()
        info = session.e.info()
        self.assertEqual(info["active_modalities"], {"audio": True, "text": True, "vision": True})
        self.assertEqual(info["response_mode"], "templates")
        self.assertFalse(info["multimodal_ready"])
        session.e.responder = BlockingResponder()
        self.assertTrue(session.e.info()["multimodal_ready"])
        session.e.text = None
        self.assertFalse(session.e.info()["multimodal_ready"])


if __name__ == "__main__":
    unittest.main()
