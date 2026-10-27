"""Short spoken reply generator: Qwen2.5-3B-Instruct (4-bit MLX) running on the Apple GPU.

The model sees the transcript, the per-modality emotion readings (voice, words, face), the
fused read, a prosody summary and a words-vs-delivery agreement note, and streams back one or
two sentences. All calls must run on the Engine's "mlx" thread.
"""
from typing import Iterator

SYSTEM_PROMPT = (
    "You are Pip, a small, warm character robot talking face to face with one person. "
    "Speak directly to them as \"you\"; never describe them as \"they\" or \"the person\". "
    "Reply with one or two short spoken sentences (under 30 words). No emojis, lists or stage directions. "
    "Answer what they said, and let how they came across set your tone. You get rough hints about their "
    "voice, words and face; each is often wrong, so never repeat them, never list emotion names, and never "
    "mention models, readings or percentages. A note appears only when their voice and face point somewhere "
    "other than their words and read more strongly (for example 'great' said angrily, or with a frown); then "
    "gently respond to how they came across rather than the literal words, without insisting you know how "
    "they feel. Without a note, go with what they said. If the transcript was garbled, say you didn't "
    "quite catch that."
)

TEMPLATES = {
    "anger": "That sounds really frustrating. Do you want to tell me what happened?",
    "disgust": "Ugh, that does not sound pleasant at all.",
    "fear": "That sounds worrying. I'm here, take your time.",
    "joy": "That's lovely to hear! Tell me more.",
    "neutral": "I see. Go on, I'm listening.",
    "sadness": "I'm sorry, that sounds hard. Do you want to talk about it?",
    "surprise": "Whoa, really? I didn't see that coming either!",
}


# Purpose: Format a reading's strongest class probabilities for a prompt.
# Input: r: reading with probs; k: number of classes to include.
# Output: Comma-separated emotion/probability text.
def _top(r, k=2):
    probs = sorted(r["probs"].items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{e} {p:.0%}" for e, p in probs)


# Purpose: Quote the transcript for the prompt, or describe it when it carries no usable words.
# Input: transcript: Whisper text; repetitive: True when Whisper's compression-ratio check failed.
# Output: Quoted transcript, or a short description of unclear or missing speech.
def _said(transcript: str, repetitive: bool = False) -> str:
    if repetitive:
        return "(unclear speech; the transcript was garbled)"
    return f'"{transcript}"' if transcript else "(no clear words, maybe a laugh, sigh or noise)"


# Purpose: Ground the robot's next reply in recognized words and multimodal state.
# Input: state: final emotion/transcript/prosody/agreement data; history: prior role/content messages.
# Output: Chat message list containing system instructions, retained history, and the current input.
def build_messages(state: dict, history: list) -> list:
    said = _said(state["transcript"], (state.get("asr") or {}).get("repetitive", False))
    mods = state.get("modalities") or {"voice": {"probs": state["probs"]}}
    lines = [f"You heard: {said}", "Hints about how they came across (often wrong):"]
    lines.append(f"- voice: {_top(mods['voice'])}")
    if mods.get("words"):
        lines.append(f"- words: {_top(mods['words'])}")
    if "face" in mods:
        lines.append(f"- face: {_top(mods['face'])}" if mods["face"] else "- face: not visible")
    lines.append(f"- overall read: {_top(state, 1)}")
    lines.append(f"- delivery: {state['delivery']}")
    agree = state.get("agreement") or {}
    if agree.get("kind") in ("conflict", "tone_only", "words_only", "differ"):
        lines.append(f"- note: {agree['note']}")
    lines.append("Your reply, spoken to them:")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        said = _said(turn["transcript"], turn.get("repetitive", False))
        msgs.append({"role": "user", "content": f"{said} (came across as {turn['emotion']})"})
        msgs.append({"role": "assistant", "content": turn["reply"]})
    msgs.append({"role": "user", "content": "\n".join(lines)})
    return msgs


class Responder:
    # Purpose: Configure the MLX language model and response length.
    # Input: repo: model repository or directory; max_tokens: default output token limit.
    # Output: None; stores generation settings without loading the model.
    def __init__(self, repo: str, max_tokens: int = 60):
        self.repo = repo
        self.max_tokens = max_tokens
        self.model = self.tokenizer = None

    # Purpose: Load the MLX model/tokenizer and warm up generation.
    # Input: None explicitly; uses this instance's repository on the Engine MLX worker.
    # Output: None; initializes the model and tokenizer for streaming.
    def load(self):
        from mlx_lm import load

        self.model, self.tokenizer = load(self.repo)
        for _ in self.stream([{"role": "user", "content": "Say hi."}], max_tokens=2):  # warm-up
            pass

    # Purpose: Generate a reply on the MLX worker and expose visible text as it arrives.
    # Input: messages: chat role/content dicts; max_tokens: optional output limit; call load() first.
    # Output: Generator yielding nonempty text deltas.
    def stream(self, messages: list, max_tokens: int = None) -> Iterator[str]:
        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens or self.max_tokens):
            text = getattr(r, "text", r)  # GenerationResponse in mlx-lm >= 0.19
            if text:
                yield text



class TemplateResponder:
    """Fallback when USE_LLM=0 or the LLM fails: canned reply per emotion (debug only)."""
    repo = "templates"

    # Purpose: Provide the responder loading interface for the fixed-template fallback.
    # Input: None.
    # Output: None; no learned model is loaded.
    def load(self):
        pass

    # Purpose: Emit an emotion-specific template when the language model is unavailable.
    # Input: messages: last user content supplies the overall emotion; max_tokens: accepted but unused.
    # Output: Generator yielding the chosen fallback template one word at a time.
    def stream(self, messages: list, max_tokens: int = None) -> Iterator[str]:
        content = messages[-1]["content"]
        emo = content.split("overall read: ")[1].split(" ")[0] if "overall read: " in content else "neutral"
        for word in TEMPLATES.get(emo, TEMPLATES["neutral"]).split(" "):
            yield word + " "
