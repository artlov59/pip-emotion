"""Transport regressions; no models, downloads, GPU or HTTP client required."""
import unittest

import numpy as np

from app.protocol import handle_message


class FakeSession:
    # Purpose: Create protocol packet/event recording state.
    # Input: None.
    # Output: None; initializes audio/frame/event lists and a turn counter.
    def __init__(self):
        self.events, self.audio, self.frames = [], [], []
        self.turn_id = 0

    # Purpose: Record a protocol event for assertions.
    # Input: event: event dict.
    # Output: None; appends to the event list.
    def emit(self, event):
        self.events.append(event)

    # Purpose: Record decoded PCM delivered by the protocol.
    # Input: samples: audio array.
    # Output: None; appends to the audio list.
    def feed(self, samples):
        self.audio.append(samples)

    # Purpose: Record JPEG packets delivered by the protocol.
    # Input: frame: encoded frame bytes.
    # Output: None; appends to the frame list.
    def feed_frame(self, frame):
        self.frames.append(frame)

    # Purpose: Simulate accepting a final turn during flush.
    # Input: None explicitly; uses the fixture turn counter.
    # Output: None; increments turn_id.
    def flush(self):
        self.turn_id += 1

    # Purpose: Simulate the reset acknowledgement.
    # Input: None.
    # Output: None; records a reset event.
    def reset(self):
        self.emit({"type": "reset"})


class ProtocolTests(unittest.TestCase):
    # Purpose: Verify invalid packets leave session usable.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_invalid_packets_leave_session_usable(self):
        session = FakeSession()
        bad_packets = [dict(data=b""), dict(data=b"\x01"), dict(data=b"\x01\x00"),
                       dict(data=b"\x02"), dict(data=b"\xffx"), dict(text="{"),
                       dict(text="[]"), dict(text="null"), dict(text='{"type":"bad"}')]
        for packet in bad_packets:
            with self.subTest(packet=packet):
                handle_message(session, **packet)
                self.assertEqual(session.events[-1]["where"], "protocol")
        handle_message(session, data=b"\x01\x00\x80\xff\x7f")
        np.testing.assert_allclose(session.audio[0], [-1.0, 32767 / 32768])
        self.assertEqual(len(session.events), len(bad_packets))

    # Purpose: Verify flush acknowledges last accepted turn.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_flush_acknowledges_last_accepted_turn(self):
        session = FakeSession()
        handle_message(session, text='{"type":"flush"}')
        self.assertEqual(session.events, [{"type": "flushed", "utterance_id": 1}])

    # Purpose: Verify jpeg and reset.
    # Input: None explicitly; uses test fixtures and controlled mocks.
    # Output: None; raises an assertion/error if the expected behavior regresses.
    def test_jpeg_and_reset(self):
        session = FakeSession()
        handle_message(session, data=b"\x02jpeg")
        handle_message(session, text='{"type":"reset"}')
        self.assertEqual(session.frames, [b"jpeg"])
        self.assertEqual(session.events, [{"type": "reset"}])


if __name__ == "__main__":
    unittest.main()
