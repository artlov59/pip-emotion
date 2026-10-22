"""Short spoken reply generator: Qwen2.5-3B-Instruct (4-bit MLX) running on the Apple GPU.

The model sees the transcript, the audio emotion distribution and a prosody summary,
and streams back one or two sentences. All calls must run on the Engine's "mlx" thread.
"""
from typing import Iterator

SYSTEM_PROMPT = (
    "You are Pip, a small, warm character robot having a face-to-face conversation. "
    "Reply with one or two short spoken sentences (under 30 words). No emojis, lists or stage directions. "
    "React to what the person said AND how they said it. The voice-emotion reading comes from an "
    "audio model that is often unsure: treat low-confidence readings as hints, not facts. "
    "If the words and the tone disagree (for example 'great' said angrily), gently respond to the tone. "
    "Never mention models, percentages or labels."
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


def build_messages(state: dict, history: list) -> list:
    probs = sorted(state["probs"].items(), key=lambda kv: -kv[1])[:3]
    emo = ", ".join(f"{k} {v:.0%}" for k, v in probs)
    said = f'"{state["transcript"]}"' if state["transcript"] else "(no clear words, maybe a laugh, sigh or noise)"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        msgs.append({"role": "user", "content": f'"{turn["transcript"]}" (sounded {turn["emotion"]})'})
        msgs.append({"role": "assistant", "content": turn["reply"]})
    msgs.append({"role": "user", "content": (
        f"They said: {said}\n"
        f"Voice emotion (audio model): {emo}\n"
        f"Delivery: {state['delivery']}\n"
        "Your reply:")})
    return msgs


class Responder:
    def __init__(self, repo: str, max_tokens: int = 60):
        self.repo = repo
        self.max_tokens = max_tokens
        self.model = self.tokenizer = None

    def load(self):
        from mlx_lm import load

        self.model, self.tokenizer = load(self.repo)
        for _ in self.stream([{"role": "user", "content": "Say hi."}], max_tokens=2):  # warm-up
            pass

    def stream(self, messages: list, max_tokens: int = None) -> Iterator[str]:
        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens or self.max_tokens):
            text = getattr(r, "text", r)  # GenerationResponse in mlx-lm >= 0.19
            if text:
                yield text

    def num_parameters(self) -> int:
        from app.mlx_params import count_mlx_parameters
        return count_mlx_parameters(self.model)


class TemplateResponder:
    """Fallback when USE_LLM=0 or the LLM fails: canned reply per emotion (debug only)."""
    repo = "templates"

    def load(self):
        pass

    def stream(self, messages: list, max_tokens: int = None) -> Iterator[str]:
        content = messages[-1]["content"]
        emo = content.split("Voice emotion (audio model): ")[1].split(" ")[0] if "Voice emotion" in content else "neutral"
        for word in TEMPLATES.get(emo, TEMPLATES["neutral"]).split(" "):
            yield word + " "

    def num_parameters(self) -> int:
        return 0
