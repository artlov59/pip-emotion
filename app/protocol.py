"""Validate the small streaming input protocol independently of the web server."""
import json

import numpy as np

KIND_AUDIO, KIND_JPEG = 0x01, 0x02


# Purpose: Validate and dispatch one incoming WebSocket packet.
# Input: session: active Session; data: optional typed PCM/JPEG bytes; text: optional JSON command string.
# Output: None; feeds media or handles flush/reset, emitting errors for malformed input while retaining the
#          session.
def handle_message(session, *, data=None, text=None):
    """Consume one message; malformed input produces an error without losing the session."""
    # Purpose: Report a protocol error without terminating the conversation.
    # Input: message: readable rejection reason; captures session.
    # Output: None; emits an error event.
    def reject(message):
        session.emit({"type": "error", "where": "protocol", "message": message})

    if data is not None:
        if not data:
            reject("Binary message must include a kind byte and payload")
            return
        kind, payload = data[0], data[1:]
        if kind == KIND_AUDIO:
            if not payload or len(payload) % 2:
                reject("Audio payload must contain complete little-endian int16 samples")
                return
            session.feed(np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0)
        elif kind == KIND_JPEG:
            if not payload:
                reject("JPEG payload is empty")
                return
            session.feed_frame(payload)
        else:
            reject("Unknown binary message kind")
        return

    if text is None:
        reject("Expected binary audio/JPEG or a JSON command")
        return
    try:
        command = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        reject("Command must be valid JSON")
        return
    if not isinstance(command, dict):
        reject("Command must be a JSON object")
        return
    kind = command.get("type")
    if kind == "flush":
        session.flush()
        # An acknowledgement is ordered after all previous input has been accepted.
        # Clients can now distinguish no speech from a turn waiting for inference.
        session.emit({"type": "flushed", "utterance_id": session.turn_id})
    elif kind == "reset":
        session.reset()
    else:
        reject("Unknown command; expected flush or reset")
