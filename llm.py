from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from gsm8k_env import GSM8KExample


@dataclass(frozen=True)
class SampleResult:
    completion: str
    logprob: float
    ref_logprob: float
    completion_token_ids: Optional[list[int]] = None


class StubLLM:
    """Deterministic offline backend with simple per-example skill updates."""

    def __init__(self, examples: list[GSM8KExample], seed: int) -> None:
        self._rng = random.Random(seed)
        self._skill: dict[str, float] = {ex.example_id: -2.2 for ex in examples}
        self._reference_skill: dict[str, float] = dict(self._skill)
        self._gold: dict[str, str] = {ex.example_id: ex.gold_answer for ex in examples}
        self._global_shift: float = 0.0
        self._version = 0

    @property
    def model_version(self) -> str:
        return f"stub-v{self._version}"

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 50:
            return 1.0
        if x <= -50:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _incorrect_answer(self, gold: str) -> str:
        base = int(float(gold)) if gold.replace("-", "").isdigit() else int(self._rng.randint(1, 100))
        delta = self._rng.randint(1, 20)
        return str(base + delta)

    def sample(
        self,
        example_id: str,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
    ) -> SampleResult:
        del prompt, max_new_tokens
        scaled_skill = (self._skill[example_id] + self._global_shift) / max(temperature, 1e-3)
        p_correct = max(1e-6, min(1.0 - 1e-6, self._sigmoid(scaled_skill)))

        ref_scaled = self._reference_skill[example_id] / max(temperature, 1e-3)
        p_ref = max(1e-6, min(1.0 - 1e-6, self._sigmoid(ref_scaled)))

        is_correct = self._rng.random() < p_correct
        gold = self._gold[example_id]
        answer = gold if is_correct else self._incorrect_answer(gold)
        if self._rng.random() < 0.8:
            completion = f"Working...\n#### {answer}"
        else:
            completion = f"Answer: {answer}"

        if is_correct:
            logprob = math.log(p_correct)
            ref_logprob = math.log(p_ref)
        else:
            logprob = math.log(1.0 - p_correct)
            ref_logprob = math.log(1.0 - p_ref)

        return SampleResult(completion=completion, logprob=logprob, ref_logprob=ref_logprob)

    def apply_group_update(
        self,
        example_id: str,
        prompt: str,
        group_rows: list[dict[str, Any]],
        weights: list[float],
        lr: float,
        lambda_kl: float = 0.1,
    ) -> None:
        del prompt, lambda_kl
        rewards = [float(r["reward"]) for r in group_rows]
        signal = sum(w * r for w, r in zip(weights, rewards))
        self._skill[example_id] += lr * signal * 4.0
        self._global_shift += lr * signal * 0.4
        self._version += 1


class MlxBackend:
    """Real mlx-lm backend with LoRA policy updates and frozen reference policy."""

    def __init__(
        self,
        model_name_or_path: str,
        seed: int,
        lora_rank: int = 32,
        lora_num_layers: int = -1,
        trust_remote_code: bool = True,
        init_lr: float = 1e-4,
        debug: bool = False,
    ) -> None:
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import load as mlx_load
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "mlx backend requested but dependencies are unavailable. "
                "Install mlx and mlx-lm, or use --backend stub."
            ) from exc

        self.mx = mx
        self.nn = nn
        self.stream_generate = stream_generate
        self.make_sampler = make_sampler
        self._debug = debug
        if self._debug:
            print("[DEBUG] mlx backend: loading policy model...", flush=True)

        tokenizer_cfg = {"trust_remote_code": trust_remote_code}
        self.model, self.tokenizer = mlx_load(
            model_name_or_path,
            tokenizer_config=tokenizer_cfg,
        )
        if self._debug:
            print("[DEBUG] mlx backend: loading reference model...", flush=True)
        self.ref_model, _ = mlx_load(
            model_name_or_path,
            tokenizer_config=tokenizer_cfg,
        )

        self.model.freeze()
        if self._debug:
            print(f"[DEBUG] mlx backend: applying LoRA rank={lora_rank}...", flush=True)
        linear_to_lora_layers(
            self.model,
            lora_num_layers,
            {"rank": lora_rank, "scale": 20.0, "dropout": 0.0},
        )
        self.ref_model.freeze()
        self.model.eval()
        self.ref_model.eval()

        self.optimizer = optim.Adam(learning_rate=init_lr, betas=(0.9, 0.95), eps=1e-8)
        self._loss_and_grad = nn.value_and_grad(self.model, self._weighted_loss)
        self._seed = seed
        self._sample_calls = 0
        self._version = 0
        if self._debug:
            print("[DEBUG] mlx backend: initialization complete.", flush=True)

    @property
    def model_version(self) -> str:
        return f"mlx-v{self._version}"

    def _encode_prompt(self, prompt: str) -> list[int]:
        bos_token = getattr(self.tokenizer, "bos_token", None)
        add_special_tokens = bos_token is None or not prompt.startswith(bos_token)
        return list(self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens))

    def _completion_logprob(
        self,
        model: Any,
        prompt_ids: list[int],
        completion_ids: list[int],
    ) -> float:
        if not completion_ids:
            return 0.0
        t0 = time.perf_counter()
        full_ids = prompt_ids + completion_ids
        tokens = self.mx.array(full_ids, dtype=self.mx.int32)[None, :]
        logits = model(tokens[:, :-1])
        log_probs = self.nn.log_softmax(logits.astype(self.mx.float32), axis=-1)
        targets = tokens[:, 1:]
        token_lp = self.mx.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        start = len(prompt_ids) - 1
        end = start + len(completion_ids)
        completion_lp = token_lp[:, start:end]
        total = self.mx.sum(completion_lp)
        self.mx.eval(total)
        if self._debug:
            dt = time.perf_counter() - t0
            print(
                f"[DEBUG] mlx ref_logprob pass: prompt_tokens={len(prompt_ids)} "
                f"completion_tokens={len(completion_ids)} took={dt:.3f}s",
                flush=True,
            )
        return float(total.item())

    def sample(
        self,
        example_id: str,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
    ) -> SampleResult:
        del example_id
        self.mx.random.seed(self._seed + self._sample_calls)
        self._sample_calls += 1
        sample_id = self._sample_calls
        if self._debug:
            print(
                f"[DEBUG] mlx sample call={sample_id} temp={temperature} "
                f"max_new_tokens={max_new_tokens} prompt_chars={len(prompt)}",
                flush=True,
            )

        t0 = time.perf_counter()
        prompt_ids = self._encode_prompt(prompt)
        if self._debug:
            print(
                f"[DEBUG] mlx sample call={sample_id}: encoded prompt_tokens={len(prompt_ids)}",
                flush=True,
            )
        sampler = self.make_sampler(temp=temperature, top_p=1.0, min_p=0.0, top_k=0)

        text_parts: list[str] = []
        completion_ids: list[int] = []
        token_logprobs: list[float] = []
        last_progress_tokens = 0
        gen_start = time.perf_counter()
        for resp in self.stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
        ):
            if resp.text:
                text_parts.append(resp.text)
            if resp.finish_reason is not None:
                break
            completion_ids.append(int(resp.token))
            token_logprobs.append(float(resp.logprobs[resp.token].item()))
            if self._debug and len(completion_ids) - last_progress_tokens >= 64:
                last_progress_tokens = len(completion_ids)
                print(
                    f"[DEBUG] mlx sample call={sample_id}: generated_tokens={len(completion_ids)} "
                    f"elapsed={time.perf_counter() - gen_start:.3f}s",
                    flush=True,
                )

        gen_dt = time.perf_counter() - gen_start
        completion = "".join(text_parts)
        logprob = float(sum(token_logprobs))
        if self._debug:
            print(
                f"[DEBUG] mlx sample call={sample_id}: generation done "
                f"completion_tokens={len(completion_ids)} took={gen_dt:.3f}s",
                flush=True,
            )
            print(
                f"[DEBUG] mlx sample call={sample_id}: starting ref_logprob computation...",
                flush=True,
            )
        ref_logprob = self._completion_logprob(self.ref_model, prompt_ids, completion_ids)
        if self._debug:
            print(
                f"[DEBUG] mlx sample call={sample_id}: done total_time={time.perf_counter() - t0:.3f}s "
                f"logprob={logprob:.3f} ref_logprob={ref_logprob:.3f}",
                flush=True,
            )
        return SampleResult(
            completion=completion,
            logprob=logprob,
            ref_logprob=ref_logprob,
            completion_token_ids=completion_ids,
        )

    def _weighted_loss(self, model: Any, batch: list[dict[str, Any]], lambda_kl: float):
        total_pg = self.mx.array(0.0, dtype=self.mx.float32)
        total_kl = self.mx.array(0.0, dtype=self.mx.float32)
        total_tokens = 0

        for row in batch:
            completion_ids = row["completion_token_ids"]
            if not completion_ids:
                continue
            prompt_ids = row["prompt_ids"]
            weight = float(row["weight"])
            ref_logprob = float(row["ref_logprob"])

            full_ids = prompt_ids + completion_ids
            tokens = self.mx.array(full_ids, dtype=self.mx.int32)[None, :]
            logits = model(tokens[:, :-1])
            log_probs = self.nn.log_softmax(logits.astype(self.mx.float32), axis=-1)
            targets = tokens[:, 1:]
            token_lp = self.mx.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
            start = len(prompt_ids) - 1
            end = start + len(completion_ids)
            completion_lp = token_lp[:, start:end]

            seq_lp = self.mx.sum(completion_lp)
            total_pg = total_pg - weight * seq_lp
            total_kl = total_kl + (seq_lp - ref_logprob)
            total_tokens += len(completion_ids)

        denom = max(total_tokens, 1)
        kl = total_kl / float(denom)
        return total_pg + (lambda_kl * kl), self.mx.array(float(denom), dtype=self.mx.float32)

    def apply_group_update(
        self,
        example_id: str,
        prompt: str,
        group_rows: list[dict[str, Any]],
        weights: list[float],
        lr: float,
        lambda_kl: float = 0.1,
    ) -> None:
        del example_id
        prompt_ids = self._encode_prompt(prompt)
        batch = []
        for row, w in zip(group_rows, weights):
            completion_ids = row.get("completion_token_ids")
            if completion_ids is None:
                completion_ids = list(
                    self.tokenizer.encode(row["completion"], add_special_tokens=False)
                )
            batch.append(
                {
                    "prompt_ids": prompt_ids,
                    "completion_token_ids": completion_ids,
                    "weight": w,
                    "ref_logprob": row["ref_logprob"],
                }
            )

        self.optimizer.learning_rate = lr
        if self._debug:
            print(
                f"[DEBUG] mlx update: batch={len(batch)} lr={lr} lambda_kl={lambda_kl}",
                flush=True,
            )
        t0 = time.perf_counter()
        self.model.train()
        (_, _), grad = self._loss_and_grad(self.model, batch, lambda_kl)
        self.optimizer.update(self.model, grad)
        self.model.eval()
        self.mx.eval(self.model.parameters(), self.optimizer.state)
        self._version += 1
        if self._debug:
            print(
                f"[DEBUG] mlx update: completed in {time.perf_counter() - t0:.3f}s "
                f"new_version={self._version}",
                flush=True,
            )


def create_backend(
    backend_name: str,
    examples: list[GSM8KExample],
    seed: int,
    model_name_or_path: str = "stub",
    lora_rank: int = 32,
    trust_remote_code: bool = True,
    init_lr: float = 1e-4,
    debug: bool = False,
):
    if backend_name == "stub":
        return StubLLM(examples=examples, seed=seed)
    if backend_name == "mlx":
        if model_name_or_path in {"", "stub"}:
            raise ValueError("Provide a real model path/repo for mlx backend.")
        return MlxBackend(
            model_name_or_path=model_name_or_path,
            seed=seed,
            lora_rank=lora_rank,
            trust_remote_code=trust_remote_code,
            init_lr=init_lr,
            debug=debug,
        )
    raise ValueError(f"Unknown backend: {backend_name}")
