"""CUDA / transformers worker for AWS Ubuntu (NVIDIA GPU)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from config import CUDA_MODEL_ID, MAX_TOKENS, TEMPERATURE
from decision_parse import fallback, parse_output
from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import AgentInput, Decision

logger = logging.getLogger(__name__)


class CudaWorker:
    """One in-process Hugging Face CausalLM on CUDA."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or os.environ.get("SLM_MODEL_ID", CUDA_MODEL_ID)
        self._model = None
        self._tokenizer = None
        self._ready = False
        self._lock = threading.Lock()
        self._device = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Download (if needed) and load the model onto CUDA."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = os.environ.get("SLM_MODEL_ID", self.model_id)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This worker is for AWS Ubuntu GPU instances."
            )

        logger.info(
            "Loading CUDA model %s on %s …",
            self.model_id,
            torch.cuda.get_device_name(0),
        )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._ready = True
        logger.info("CUDA model ready: %s (%s)", self.model_id, self._device)

    def decide_one(self, agent: AgentInput) -> Decision:
        """Run inference for one agent; never raise — always return a Decision."""
        try:
            raw = self._generate(agent)
            intent, destination = parse_output(raw, agent)
            return Decision(id=agent.id, intent=intent, destination=destination)
        except Exception:
            logger.exception("Inference failed for agent %s; using fallback", agent.id)
            return fallback(agent)

    def _generate(self, agent: AgentInput) -> str:
        import torch

        if not self._ready or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not ready")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(agent)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": MAX_TOKENS,
            "do_sample": TEMPERATURE > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if TEMPERATURE > 0:
            gen_kwargs["temperature"] = TEMPERATURE

        with self._lock:
            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0, prompt_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


worker = CudaWorker()
