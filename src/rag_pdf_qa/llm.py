"""Local chat LLM via Hugging Face transformers (default: Qwen2.5-1.5B-Instruct).

Anything callable as ``llm(messages) -> str`` can be passed to ``RAGPipeline`` instead,
which is how the tests swap in a fake model.
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings, resolve_device

Message = dict[str, str]


class ChatLLM(Protocol):
    def __call__(self, messages: list[Message]) -> str: ...


class HFChatLLM:
    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(settings.device)
        self.max_new_tokens = settings.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(settings.llm_model, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self._torch = torch

    def __call__(self, messages: list[Message]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic answers for a QA tool
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
