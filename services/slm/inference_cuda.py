"""CUDA / transformers worker for AWS Ubuntu (NVIDIA GPU)."""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence

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
        # Prefer native transformers weights (Phi-4 since ~4.49). Fall back to
        # remote code only if needed; remote modeling_phi3 breaks on transformers>=4.54.
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=False,
            )
        except Exception as exc:
            logger.warning(
                "Native load failed (%s); retrying with trust_remote_code=True",
                exc,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        # Left-pad so batched generate aligns new tokens on the right.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._ready = True
        logger.info("CUDA model ready: %s (%s)", self.model_id, self._device)

    def decide_one(self, agent: AgentInput) -> Decision:
        """Run inference for one agent; never raise — always return a Decision."""
        return self.decide_many([agent])[0]

    def decide_many(self, agents: Sequence[AgentInput]) -> List[Decision]:
        """Batched inference: one generate() over independent per-agent prompts.

        Each batch row is a single-agent chat (no cross-agent tokens). Attention
        does not mix rows, so contexts stay isolated.
        """
        if not agents:
            return []
        try:
            raws = self._generate_many(agents)
        except Exception:
            logger.exception(
                "Batched inference failed for %d agents; per-agent fallback",
                len(agents),
            )
            return [fallback(a) for a in agents]

        decisions: List[Decision] = []
        for agent, raw in zip(agents, raws):
            try:
                intent, destination = parse_output(raw, agent)
                decisions.append(
                    Decision(id=agent.id, intent=intent, destination=destination)
                )
            except Exception:
                logger.exception("Parse failed for agent %s; using fallback", agent.id)
                decisions.append(fallback(agent))
        return decisions

    def _prompt_for(self, agent: AgentInput) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(agent)},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, agent: AgentInput) -> str:
        return self._generate_many([agent])[0]

    def _generate_many(self, agents: Sequence[AgentInput]) -> List[str]:
        import torch

        if not self._ready or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not ready")

        prompts = [self._prompt_for(agent) for agent in agents]
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        # Tiny JSON intents; keep this low so the batch isn't held by a long tail.
        max_new = min(MAX_TOKENS, int(os.environ.get("SLM_BATCH_MAX_NEW", "48")))

        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": TEMPERATURE > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            # Prefer stopping once the JSON object closes (transformers>=4.42).
            "stop_strings": ["}"],
            "tokenizer": self._tokenizer,
        }
        if TEMPERATURE > 0:
            gen_kwargs["temperature"] = TEMPERATURE

        with self._lock:
            with torch.inference_mode():
                try:
                    output_ids = self._model.generate(**inputs, **gen_kwargs)
                except TypeError:
                    # Older transformers without stop_strings
                    gen_kwargs.pop("stop_strings", None)
                    gen_kwargs.pop("tokenizer", None)
                    output_ids = self._model.generate(**inputs, **gen_kwargs)

        texts: List[str] = []
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, prompt_len:]
            texts.append(
                self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            )
        return texts


worker = CudaWorker()
