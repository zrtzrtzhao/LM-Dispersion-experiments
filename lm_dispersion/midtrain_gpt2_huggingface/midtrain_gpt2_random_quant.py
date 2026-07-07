"""
Mid-train GPT-2 with Random Rotation + Quantization auxiliary prediction.

The auxiliary branch rotates hidden states by a fixed random orthogonal matrix,
applies L2-preserving sign quantization in that random coordinate system, rotates
back, and asks the LM head to keep predicting the original next-token labels.
"""

from typing import List
import os
import gc
import sys
import json
import math
import argparse
import time
import torch
import torch.distributed as dist
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, os.path.join(import_dir))
from dispersion import DispersionLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console
    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
                o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')

def filter_non_empty(example):
    txt = example.get("text", "")
    return bool(txt and txt.strip())

def tokenize_batch(examples, tokenizer):
    return tokenizer(examples["text"])

def group_texts(examples, context_len):
    concatenated = {}
    for k in examples.keys():
        all_vals = []
        for seq in examples[k]:
            all_vals.extend(seq)
        concatenated[k] = all_vals
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // context_len) * context_len
    result = {k: [t[i:i+context_len] for i in range(0, total_len, context_len)]
              for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result

def compute_precision_flags():
    if not torch.cuda.is_available():
        return False, False
    if torch.cuda.is_bf16_supported():
        return False, True
    else:
        return True, False

def make_splits(dataset_name, dataset_config, cache_dir, hf_token, tokenizer, context_len, seed):
    if dataset_config is None or str(dataset_config).strip() == "":
        ds = load_dataset(dataset_name, streaming=False, token=hf_token, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_name, dataset_config, streaming=False, token=hf_token, cache_dir=cache_dir)

    if "train" in ds:
        ds_train = ds["train"]
    else:
        parts = [s for s in ("validation", "test") if s in ds]
        assert parts, "No 'train' split and no alternative splits available."
        ds_train = concatenate_datasets([ds[s] for s in parts])

    ds_train = ds_train.filter(filter_non_empty)

    if "validation" in ds:
        ds_val = ds["validation"].filter(filter_non_empty)
    elif "test" in ds:
        ds_val = ds["test"].filter(filter_non_empty)
    else:
        ds_val = ds_train

    tok_train = ds_train.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=[c for c in ds_train.column_names if c != "text"],
        desc="Tokenizing train",
    )
    tok_val = ds_val.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=[c for c in ds_val.column_names if c != "text"],
        desc="Tokenizing val",
    )

    lm_train = tok_train.map(
        lambda b: group_texts(b, context_len),
        batched=True,
        desc=f"Grouping train into blocks of {context_len}",
    ).shuffle(seed=seed)

    lm_val = tok_val.map(
        lambda b: group_texts(b, context_len),
        batched=True,
        desc=f"Grouping val into blocks of {context_len}",
    )

    return lm_train, lm_val

class LMEvalCallback(TrainerCallback):
    def __init__(self,
                 tokenizer,
                 zeroshot_tasks, fewshot_tasks,
                 log_path,
                 max_gen_tokens,
                 num_fewshot,
                 max_eval_samples=None,
                 eval_at_begin=True, eval_at_end=True,
                 every_n_steps=None, save_on_eval=True):
        self.tok = tokenizer
        self.zeroshot_tasks = zeroshot_tasks
        self.fewshot_tasks = fewshot_tasks
        self.log_path = log_path
        self.max_gen_tokens = max_gen_tokens
        self.num_fewshot = num_fewshot
        self.max_eval_samples = max_eval_samples
        self.eval_at_begin = eval_at_begin
        self.eval_at_end = eval_at_end
        self.every_n_steps = every_n_steps
        self.save_on_eval = save_on_eval
        self.has_run_begin = False
        self.eval_wall_seconds = 0.0

    def _run_evaluation(self, args, state, model, stage=""):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = (world_size > 1 and dist.is_available() and dist.is_initialized())

        if ddp:
            dist.barrier()

        try:
            if local_rank == 0:
                _lm_eval_t0 = time.perf_counter()
                try:
                    stage_str = f" ({stage})" if stage else ""
                    # Determine device configuration
                    if torch.cuda.is_available() and world_size > 1:
                        # Multi-GPU
                        device_str = "cuda"
                    elif torch.cuda.is_available():
                        # Single-GPU
                        device = next(model.parameters()).device
                        device_str = f"cuda:{device.index}" if device.index is not None else "cuda:0"
                    else:
                        # CPU
                        device_str = "cpu"

                    log(f"[LMEval] Running evaluation{stage_str} at step {state.global_step} (world_size={world_size}, device={device_str})...", filepath=self.log_path)

                    eval_model = model.module if hasattr(model, "module") else model

                    was_training = eval_model.training
                    eval_model.eval()

                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        gc.collect()

                        with torch.inference_mode():
                            wrapped_model = HFLM(pretrained=eval_model, tokenizer=self.tok, batch_size=1)

                            res_zeroshot = simple_evaluate(
                                model=wrapped_model,
                                tasks=self.zeroshot_tasks,
                                num_fewshot=0,
                                device=device_str,
                                limit=self.max_eval_samples,
                                gen_kwargs = {"max_gen_toks": self.max_gen_tokens, "do_sample": False},
                                log_samples=False,  # Otherwise, will log individual samples in the JSON.
                                random_seed=args.seed,
                                numpy_random_seed=args.seed,
                                torch_random_seed=args.seed,
                                fewshot_random_seed=args.seed,
                            )

                            res_fewshot = simple_evaluate(
                                model=wrapped_model,
                                tasks=self.fewshot_tasks,
                                num_fewshot=self.num_fewshot,
                                device=device_str,
                                limit=self.max_eval_samples,
                                gen_kwargs = {"max_gen_toks": self.max_gen_tokens, "do_sample": False},
                                log_samples=False,  # Otherwise, will log individual samples in the JSON.
                                random_seed=args.seed,
                                numpy_random_seed=args.seed,
                                torch_random_seed=args.seed,
                                fewshot_random_seed=args.seed,
                            )

                        assert "results" in res_zeroshot and "results" in res_fewshot
                        filename = f"lm_eval_{stage}_{state.global_step}.json" if stage else f"lm_eval_step{state.global_step}.json"
                        out = os.path.join(args.output_dir, filename)
                        merged_dict = {**res_zeroshot["results"], **res_fewshot["results"]}
                        with open(out, "w") as f:
                            json.dump({"results": merged_dict}, f, indent=2)
                        log(f"[LMEval] Results saved to {out}", filepath=self.log_path)

                        for task, metrics in merged_dict.items():
                            if isinstance(metrics, dict):
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        log(f"[LMEval] {task}.{metric_name}: {value:.4f}", filepath=self.log_path)

                        if self.save_on_eval:
                            ckpt_dir = os.path.join(args.output_dir, f"eval_ckpt_{stage or 'interval'}_step{state.global_step}")
                            os.makedirs(ckpt_dir, exist_ok=True)
                            if hasattr(model, 'module'):
                                model.module.save_pretrained(ckpt_dir, save_safetensors=getattr(args, "save_safetensors", True))
                            else:
                                model.save_pretrained(ckpt_dir, save_safetensors=getattr(args, "save_safetensors", True))
                            self.tok.save_pretrained(ckpt_dir)
                            log(f"[LMEval] Weights saved to {ckpt_dir}", filepath=self.log_path)

                    except Exception as e:
                        log(f"[LMEval] Error during evaluation{stage_str} at step {state.global_step}: {e}", filepath=self.log_path)

                    finally:
                        if was_training:
                            eval_model.train()
                finally:
                    self.eval_wall_seconds += time.perf_counter() - _lm_eval_t0

        finally:
            if ddp:
                dist.barrier()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_at_begin and not self.has_run_begin:
            model = kwargs["model"]
            self._run_evaluation(args, state, model, "begin")
            self.has_run_begin = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps is None:
            return

        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        model = kwargs["model"]
        self._run_evaluation(args, state, model, "interval")

    def on_train_end(self, args, state, control, **kwargs):
        if self.eval_at_end:
            model = kwargs["model"]
            self._run_evaluation(args, state, model, "end")


class CausalLMLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, seq_len, V], labels: [B, seq_len]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss


class CustomLossTrainer(Trainer):
    def __init__(self,
                 *args,
                 loss_fn: torch.nn.Module,
                 dispersion: str,
                 dispersion_coeff: float,
                 dispersion_loc: str,
                 tau_l2: float,
                 tau_cos: float,
                 clamp_threshold: float,
                 rq: bool,
                 rq_coeff: float,
                 rq_loc: str,
                 rq_type: str,
                 rq_basis: str,
                 rq_refresh: str,
                 rq_epsilon: float,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

        self.use_disp = dispersion is not None and dispersion_coeff > 0.0
        self.disp_coeff = dispersion_coeff
        self.disp_loc = dispersion_loc

        if self.use_disp:
            variant = dispersion.lower()
            self.disp_loss_fn = DispersionLoss(variant=variant,
                                               tau_l2=tau_l2,
                                               tau_cos=tau_cos,
                                               clamp_threshold=clamp_threshold)

        self.use_rq = bool(rq) and rq_coeff > 0.0
        self.rq_coeff = float(rq_coeff)
        self.rq_loc = rq_loc
        self.rq_type = rq_type
        self.rq_basis = rq_basis
        self.rq_refresh = rq_refresh
        self.rq_epsilon = float(rq_epsilon)
        self._rq_basis_cache = {}

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    def _select_transformer_layers(self, hidden_states: List[torch.Tensor], loc: str) -> List[int]:
        if loc == "last":
            return [-1]

        assert len(hidden_states) > 1
        # Transformer block outputs: indices 1 .. len-1 (skip embedding at 0)
        tr_indices = list(range(1, len(hidden_states)))
        n_tr = len(tr_indices)
        mid = n_tr // 2

        if loc == "early_half":
            return tr_indices[:mid] if mid > 0 else tr_indices[:1]
        if loc == "late_half":
            return tr_indices[mid:] if mid < n_tr else tr_indices[-1:]
        return tr_indices

    def _get_rq_basis(self, dim: int, device: torch.device) -> torch.Tensor:
        if self.rq_basis != "fixed":
            raise ValueError(f"Unsupported rq_basis: {self.rq_basis}")

        device_key = (device.type, device.index if device.index is not None else -1)
        cache_key = (dim, device_key)
        use_cache = self.rq_refresh == "never"
        if use_cache and cache_key in self._rq_basis_cache:
            return self._rq_basis_cache[cache_key]

        step = int(getattr(self.state, "global_step", 0)) if self.rq_refresh == "step" else 0
        seed = int(self.args.seed) + 1009 * dim + step
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        mat = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
        q, r = torch.linalg.qr(mat, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        q = q.to(device=device)

        if use_cache:
            self._rq_basis_cache[cache_key] = q
        return q

    def _random_rotate_quantize(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.size(-1) < 2:
            return hidden

        orig_dtype = hidden.dtype
        h = hidden.float()
        dim = h.size(-1)
        basis = self._get_rq_basis(dim, h.device)
        rotated = torch.matmul(h, basis)

        if self.rq_type != "scaled_sign":
            raise ValueError(f"Unsupported rq_type: {self.rq_type}")

        sign = torch.where(rotated >= 0, torch.ones_like(rotated), -torch.ones_like(rotated))
        scale = rotated.norm(p=2, dim=-1, keepdim=True).clamp_min(self.rq_epsilon) / math.sqrt(dim)
        quantized = sign * scale

        # Straight-through estimator: forward uses quantized values, backward follows rotated.
        quantized_ste = rotated + (quantized - rotated).detach()
        restored = torch.matmul(quantized_ste, basis.transpose(0, 1))
        return restored.to(dtype=orig_dtype)

    def _lm_head_logits(self, model, hidden: torch.Tensor) -> torch.Tensor:
        unwrapped = self._unwrap_model(model)
        lm_head = unwrapped.get_output_embeddings()
        if lm_head is None:
            lm_head = getattr(unwrapped, "lm_head", None)
        if lm_head is None:
            raise ValueError("Cannot find LM head via get_output_embeddings() or model.lm_head.")
        logits = lm_head(hidden)
        final_bias = getattr(unwrapped, "final_logits_bias", None)
        if final_bias is not None:
            logits = logits + final_bias
        return logits

    def random_quant_hidden_states(
        self,
        model,
        hidden_states: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loc = self.rq_loc.lower()
        sel = self._select_transformer_layers(hidden_states, loc)

        loss_values = []
        for idx in sel:
            rq_hidden = self._random_rotate_quantize(hidden_states[idx])
            rq_logits = self._lm_head_logits(model, rq_hidden)
            loss_values.append(self.loss_fn(rq_logits, labels))
        return torch.stack(loss_values).mean()

    def disperse_hidden_states(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        '''
        Computes dispersion for last layer or averages across transformer layers (excluding emb at index 0).
        hidden_states: tuple of tensors, each [B, seq_len, feature_dim]
        '''
        loc = self.disp_loc.lower()
        sel = self._select_transformer_layers(hidden_states, loc)

        loss_values = [self.disp_loss_fn(hidden_states[i]) for i in sel]
        return torch.stack(loss_values).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        # Use hidden states ONLY if we're training AND a geometry objective is on.
        want_disp = self.use_disp and model.training
        want_rq = self.use_rq and model.training
        want_hidden = want_disp or want_rq
        outputs = model(**inputs, output_hidden_states=want_hidden)
        logits = outputs.logits

        default_loss = self.loss_fn(logits, labels)
        total_loss = default_loss

        # Add dispersion ONLY in training
        if want_disp:
            disp_loss = self.disperse_hidden_states(outputs.hidden_states)
            total_loss = total_loss + self.disp_coeff * disp_loss
            outputs.dispersion_loss = disp_loss.detach()
        else:
            disp_loss = torch.zeros_like(default_loss)

        if want_rq:
            rq_loss = self.random_quant_hidden_states(
                model,
                outputs.hidden_states,
                labels=labels,
            )
            total_loss = total_loss + self.rq_coeff * rq_loss
            outputs.rq_loss = rq_loss.detach()
        else:
            rq_loss = torch.zeros_like(default_loss)

        if (model.training and
            self.state.global_step > 0 and
            self.state.global_step % self.args.logging_steps == 0):

            custom_losses = {
                "train/dispersion_loss": disp_loss.detach().item(),
                "train/rq_loss": rq_loss.detach().item(),
                "train/default_loss": default_loss.detach().item(),
                "train/total_loss": total_loss.detach().item(),
            }

            # Log to trainer's system
            self.log(custom_losses)

        return (total_loss, outputs) if return_outputs else total_loss


def main(args):
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    out_dir = args.output_dir
    if os.path.isdir(out_dir) and any(
        name.startswith("lm_eval_") and name.endswith(".json") for name in os.listdir(out_dir)
    ):
        log(
            f"Skipping training: {out_dir} already exists with lm_eval_*.json results.",
            filepath=args.log_path,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token, cache_dir=args.cache_dir)
    tokenizer.padding_side = "right"  # During (batched) training, pad to the right.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(args.model_name, token=args.hf_token, cache_dir=args.cache_dir)
    if hasattr(config, "loss_type"):
        delattr(config, "loss_type")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, token=args.hf_token, cache_dir=args.cache_dir)
    model.gradient_checkpointing_enable()

    max_position_embeddings = getattr(model.config, "max_position_embeddings")
    context_len = 1024
    max_gen_tokens = 256
    assert max_gen_tokens <= context_len and context_len <= max_position_embeddings
    tokenizer.model_max_length = context_len

    # vocab_size = len(tokenizer)
    # model.resize_token_embeddings(vocab_size)
    # model.config.vocab_size = vocab_size
    # if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
    #     model.base_model.config.vocab_size = vocab_size

    if args.lora:
        log("Applying LoRA configuration...", filepath=args.log_path)
        model.config.use_cache = False
        lora_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj", "c_fc"],  # for GPT-2, adjust for other models
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        log(f"LoRA applied. Trainable parameters: {model.get_nb_trainable_parameters()}", filepath=args.log_path)

    lm_train, lm_val = make_splits(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        tokenizer=tokenizer,
        context_len=context_len,
        seed=args.seed,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = args.per_device_train_batch_size * context_len * args.gradient_accumulation_steps * world_size
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step computed as 0; check batch size/accumulation/context length.")
    max_steps = math.ceil(args.train_tokens / tokens_per_step)
    log(f"Training for {args.train_tokens} tokens, which is {max_steps} steps.", filepath=args.log_path)
    log_every_n_steps = max_steps // args.num_ckpt + 1

    fp16, bf16 = compute_precision_flags()
    # args.lr is set assuming world size is 1.
    learning_rate = args.lr * math.sqrt(world_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_steps=max_steps,
        warmup_ratio=0.2,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        log_level="info",
        logging_steps=max(1, max_steps // 20),
        log_on_each_node=False,
        save_strategy="no",  # We will save checkpoints using LMEvalCallback.
        report_to="none",
        seed=args.seed,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        loss_fn=CausalLMLoss(),
        dispersion=args.dispersion,
        dispersion_coeff=args.dispersion_coeff,
        dispersion_loc=args.dispersion_loc,
        tau_cos=args.tau_cos,
        tau_l2=args.tau_l2,
        clamp_threshold=args.clamp_threshold,
        rq=args.rq,
        rq_coeff=args.rq_coeff,
        rq_loc=args.rq_loc,
        rq_type=args.rq_type,
        rq_basis=args.rq_basis,
        rq_refresh=args.rq_refresh,
        rq_epsilon=args.rq_epsilon,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    log("=== Mid-training setup ===", filepath=args.log_path)
    log(f"Model: {args.model_name}", filepath=args.log_path)
    log(str(model.config), filepath=args.log_path)
    log(f"Dataset: {args.dataset_name} ({args.dataset_config})", filepath=args.log_path)
    log(f"Context length: {context_len}", filepath=args.log_path)
    log(f"Max gen tokens: {max_gen_tokens}", filepath=args.log_path)
    log(f"Per-device batch: {args.per_device_train_batch_size} | Grad accum: {args.gradient_accumulation_steps} | World size: {world_size}", filepath=args.log_path)
    log(f"Token budget: {args.train_tokens} | Tokens/step: {tokens_per_step} | Max steps: {max_steps}", filepath=args.log_path)
    log(f"Precision: {'bf16' if bf16 else ('fp16' if fp16 else 'fp32')}", filepath=args.log_path)
    log(
        f"Random Rotation + Quantization: enabled={args.rq} | coeff={args.rq_coeff} | "
        f"loc={args.rq_loc} | type={args.rq_type} | basis={args.rq_basis} | refresh={args.rq_refresh}",
        filepath=args.log_path,
    )

    # https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
    zeroshot_tasks = [
        "anli",
        "hellaswag",
        "lambada",
        "openbookqa",
        "paloma_wikitext_103",
        "piqa",
        # "squad_completion",
        "truthfulqa_mc2",
        "winogrande",
    ]
    fewshot_tasks = [
        "arc_challenge",
        "arc_easy",
        # "drop",
        # "gsm8k",
        "mmlu",
        # "mmlu_pro",  # MMLU-Pro is too slow.
        "medmcqa",
    ]
    lm_eval_callback = LMEvalCallback(
        tokenizer,
        zeroshot_tasks,
        fewshot_tasks,
        log_path=args.log_path,
        max_gen_tokens=max_gen_tokens,
        num_fewshot=args.num_fewshot,
        max_eval_samples=args.max_eval_samples,
        every_n_steps=log_every_n_steps if args.train_tokens > 0 else None,
        eval_at_begin=args.eval_at_begin,
        eval_at_end=args.train_tokens > 0,
        save_on_eval=not args.no_save_model,
    )
    trainer.add_callback(lm_eval_callback)

    torch.cuda.reset_peak_memory_stats()

    train_t0 = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    log(f"Peak memory: {peak_gb:.2f} GB", filepath=args.log_path)

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        eval_sec = lm_eval_callback.eval_wall_seconds
        train_wo_eval = max(0.0, train_elapsed - eval_sec)
        log(
            f"Training wall time: {train_elapsed:.2f}s ({train_elapsed / 60:.2f} min, {train_elapsed / 3600:.4f} h)",
            filepath=args.log_path,
        )
        log(
            f"LMEvalCallback wall time (rank 0 eval work): {eval_sec:.2f}s ({eval_sec / 60:.2f} min, {eval_sec / 3600:.4f} h)",
            filepath=args.log_path,
        )
        log(
            f"Training wall time minus LMEvalCallback: {train_wo_eval:.2f}s ({train_wo_eval / 60:.2f} min, {train_wo_eval / 3600:.4f} h)",
            filepath=args.log_path,
        )

    log(f"Done. Saved to {args.output_dir}", filepath=args.log_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mid-train GPT-2 with Random Rotation + Quantization.")
    ap.add_argument("--model_name", type=str, default="gpt2",
                    help="Hugging Face model id to start from (pretrained).")
    ap.add_argument("--lora", action="store_true", help="Use LoRA (Low-Rank Adaptation) instead of full fine-tuning")
    ap.add_argument("--cache_dir", type=str, default='./.cache/')
    ap.add_argument("--dataset_name", type=str, default="Salesforce/wikitext",
                    help="Hugging Face dataset id.")
    ap.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1",
                    help="Dataset config (e.g., wikitext-2-raw-v1).")
    ap.add_argument("--hf_token", type=str, default=None,
                    help="HF token if needed for gated/private datasets.")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="Learning rate. Please set this assuming number of GPU is 1. We will scale accordingly.")
    ap.add_argument("--train_tokens", type=int, required=True,
                    help="Total number of tokens to train on (token budget).")
    ap.add_argument("--dispersion", type=str, default=None, help="Dispersion loss.")
    ap.add_argument("--dispersion_coeff", type=float, default=1, help="Dispersion loss weight.")
    ap.add_argument("--dispersion_loc", type=str, default="all", choices=["all", "last", "early_half", "late_half"])
    ap.add_argument("--tau_l2", type=float, default=1.0, help="Temperature.")
    ap.add_argument("--tau_cos", type=float, default=1.0, help="Temperature.")
    ap.add_argument("--clamp_threshold", type=float, default=0.1, help="Clamp threshold.")
    ap.add_argument("--rq", action="store_true",
                    help="Enable Random Rotation + Quantization auxiliary LM loss.")
    ap.add_argument("--rq_coeff", type=float, default=0.03,
                    help="Weight for the Random Rotation + Quantization auxiliary loss.")
    ap.add_argument("--rq_loc", type=str, default="last",
                    choices=["all", "last", "early_half", "late_half"],
                    help="Hidden layer(s) where Random Rotation + Quantization is applied.")
    ap.add_argument("--rq_type", type=str, default="scaled_sign",
                    choices=["scaled_sign"],
                    help="Quantizer type. scaled_sign preserves each token hidden vector's L2 norm.")
    ap.add_argument("--rq_basis", type=str, default="fixed",
                    choices=["fixed"],
                    help="Random orthogonal basis type.")
    ap.add_argument("--rq_refresh", type=str, default="never",
                    choices=["never", "step"],
                    help="Whether to reuse one fixed random basis or refresh it by training step.")
    ap.add_argument("--rq_epsilon", type=float, default=1e-6,
                    help="Numerical epsilon for Random Rotation + Quantization.")
    ap.add_argument("--num_fewshot", type=int, default=1, help="Eval num_fewshot.")
    ap.add_argument("--max_eval_samples", type=int, default=500, help="Eval max_eval_samples.")
    ap.add_argument("--num_ckpt", type=int, default=5, help="Number of checkpoints.")
    ap.add_argument("--no_save_model", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_at_begin", action="store_true", help="lm-eval at step 0 (slow on random init).")

    args = ap.parse_args()

    lora_suffix = "_lora" if args.lora else ""
    rq_tag = f"_rq-{args.rq_coeff}-{args.rq_loc}-{args.rq_type}-{args.rq_basis}-{args.rq_refresh}" if args.rq else "_rq-None"
    args.output_dir = f'./results/midtrain_{args.model_name}{lora_suffix}_{"-".join(args.dataset_name.split("/"))}_lr-{args.lr}_token-{args.train_tokens}_disp-{args.dispersion}-{args.dispersion_coeff}-{args.dispersion_loc}-tau_cos-{args.tau_cos}-tau_l2-{args.tau_l2}{rq_tag}_fewshot-{args.num_fewshot}_maxsample-{args.max_eval_samples}_seed-{args.seed}'
    args.log_path = os.path.join(args.output_dir, 'log.txt')
    main(args)
