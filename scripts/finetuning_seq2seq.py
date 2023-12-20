#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization and text-to-text classification
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import os
import argparse
import json
import math
import os
import random
from pprint import pformat
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nltk
import datasets
import evaluate

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset

import wandb
from tqdm.auto import tqdm, trange
from loguru import logger

import peft
import adapters
from adapters import LlamaAdapterModel, T5AdapterModel

import peft_comparison
import peft_comparison.utils
import peft_comparison.text2text_utils
import peft_comparison.mappings
from peft_comparison.collation import DataCollatorForSeq2SeqWithMetadata, DataCollatorForCausalLMWithMetadata

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

nltk.data.find("tokenizers/punkt")

str2bool = lambda x: x.lower() in ["true", "1"]


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")

    # Dataset Configuration
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use via the datasets library.")
    parser.add_argument("--task_type", default=None, choices=["summarization", "classification"])
    parser.add_argument("--dataset_config_name", default=None, help="""
                        The configuration name of the dataset to use via the datasets library
                        E.g., for superglue/glue it would be one of:
                        "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli", "boolq", "cb", "copa", "multirc"
                        And for cnn_dailymail it can be "3.0.0"
                        Lookup huggingface.co/datasets for more information.
                        """)
    parser.add_argument("--max_source_length", type=int, default=512, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--source_prefix", type=str, default="", help="A prefix to add before every source text, useful for T5 models.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--subsample_data", type=int, default=None, help="If passed, will subsample the dataset to this many examples. (debug only)")

    # Target Text Configuration
    parser.add_argument("--max_target_length", type=int, default=128, help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded during evaluate and predict.")
    parser.add_argument("--val_max_target_length", type=int, default=None, help="The maximum total sequence length for validation target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. Will default to max_target_length. This argument is also used to override the max_length param of model.generate, which is used during evaluate and predict.")
    parser.add_argument("--num_beams", type=int, default=None, help="Number of beams to use for evaluation. This argument will be passed to model.generate, which is used during evaluate and predict.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to max_length. Otherwise, dynamic padding is used.")
    parser.add_argument("--max_eval_steps_durig_validation", type=int, default=100, help="Maximum number of evaluation steps to perform during validation. Useful to save time when you don't need to run the full validation set.")

    # Model Configuration
    parser.add_argument("--model_name_or_path", type=str, default="t5-base", help="Path to pretrained model or model identifier from huggingface.co/models.")

    # Batch Configuration
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None, help="Batch size per device for the evaluation dataloader.")
    parser.add_argument("--total_batch_size", type=int, default=32, help="Total batch size per_device_batch_size * num_devices * gradient_accumulation")

    # Training Configuration
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate after the potential warmup period to use.")
    parser.add_argument("--lr_scheduler_warmup_percent", type=float, default=0.06, help="Percentage of steps for the warmup in the lr scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--min_train_steps", type=int, default=100)
    parser.add_argument("--eval_every_steps", type=int, default=None, help="Evaluate model after these many steps.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use, choices: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")

    # Output and Tracking
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")

    # PEFT Configuration
    parser.add_argument("--adapter_config_string", default=None, type=str, help="The adapter config string to use for adapter-transformers")

    # Memory Management
    parser.add_argument("--load_in_8bit", action="store_true", help="Enable 8bit quantization.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable 4bit quantization.")
    parser.add_argument("--torch_dtype", type=torch.dtype, default=torch.bfloat16, help="This sets the dtype of the remaining non quantized layers. 'bitsandbytes' library suggests to set the value to 'torch.float16' for 8 bit model and use the same dtype as the compute dtype for 4 bit model")

    # Weight and Biases Configuration
    parser.add_argument("--wandb_project", type=str, default="PEFT_Comparison", help="Name to be given to Weight and Biases logging repository")
    parser.add_argument("--tags", type=str, default=None, help="Tags to be given to individual runs in WandB repository, e.g. 'trial, t5-base, classification'")
    parser.add_argument("--wandb_name", type=str, default=None, help="Display name for the run")

    # Misc
    parser.add_argument("--verbosity", type=int, default=1, help="verbosity of the logger (1 or 2 for now)")
    parser.add_argument("--profile", action="store_true", help="Enable profiler and log profile traces to pytorch_tensorboard")
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need dataset name")

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # this allows to effectively mitigate OOM issues with beam search
    if args.per_device_eval_batch_size is None and args.num_beams < 5:
        args.per_device_eval_batch_size = args.per_device_train_batch_size

    if args.per_device_eval_batch_size is None and args.num_beams >= 5:
        args.per_device_eval_batch_size = max(1, args.per_device_train_batch_size // 2)

    if args.dataset_name == "cnn_dailymail":
        args.dataset_config_name = "3.0.0"

    if args.task_type is None:
        if args.dataset_name in peft_comparison.mappings.summarization_name_mapping:
            args.task_type = "summarization"
        elif args.dataset_config_name in peft_comparison.mappings.task_to_keys:
            args.task_type = "classification"
        else:
            raise ValueError(f"--task_type must be specified for unknown dataset name {args.dataset_name}. "
                             "But honestly, probably this dataset is not supported anyway. "
                             "To support dataset you need to at least include it into "
                             "peft_comparison.mappings.summarization_name_mapping or peft_comparison.mappings.task_to_keys "
                             "and add a postprocessing function")

    args.decoder_only = False if "t5" in args.model_name_or_path else True

    if args.output_dir == "None":  # I am not sure why this happens
        args.output_dir = None

    return args


def preprocess_adapter_config_string_for_t5(adapter_config_string, model_config):
    if "prefix_tuning" in adapter_config_string:
        if "[" in adapter_config_string:
            logger.info(f"Custom config string provided: {adapter_config_string}")
            logger.info(f"Using the custom config string as is, if you get an error related to kv_size, please check the config string")
            logger.info(f"It should contain kv_size for T5-3B and T5-11B")
            return adapter_config_string

        logger.info(f"Adding [kv_size={model_config.d_kv}] to the adapter config string")
        adapter_config_string += f"[kv_size={model_config.d_kv}]"
        logger.info(f"Using adapter config string: {adapter_config_string}")
        return adapter_config_string

    if adapter_config_string == "unipelt":
        logger.info("Building default unipelt config for T5. Non-default unipelt configuratoins need to be provided explicitly")
        _lora = "lora[r=8,use_gating=True]"
        _prefix = f"prefix_tuning[prefix_length=10,use_gating=True,kv_size={model_config.d_kv}]"
        _adapter = "seq_bn[reduction_factor=16,use_gating=True]"
        adapter_config_string = f"{_lora}|{_prefix}|{_adapter}"
        logger.info(f"Using adapter config string: {adapter_config_string}")
        return adapter_config_string

    if adapter_config_string == "mam":
        logger.info("Building default MAM config for T5. Non-default mam configuratoins need to be provided explicitly")
        _prefix = f"prefix_tuning[bottleneck_size=800,kv_size={model_config.d_kv}]"
        _adapter = "par_bn"
        adapter_config_string = f"{_prefix}|{_adapter}"
        logger.info(f"Using adapter config string: {adapter_config_string}")
        return adapter_config_string

    return adapter_config_string


def load_model_with_adapters_and_lm_head(
        *,
        hf_model_class,
        adapters_model_class,
        model_name_or_path,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=None,
    ):
    """
    This function requires Adapters branch of the Adapter-hub library:
    https://github.com/adapter-hub/adapter-transformers/tree/adapters

    """
    # @NOTE: we are using torch.float32 to overcome the compatibility issues with the new Adapters library
    # Currently testing if bf16 would work
    torch_dtype = torch_dtype or torch.float32
    constructor_kwargs = {
        "torch_dtype": torch_dtype,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
    }
    if "llama" in model_name_or_path.lower():
        constructor_kwargs["use_flash_attention_2"] = True
    logger.info(f"Loading model with kwargs {pformat(constructor_kwargs)}")

    # first we load the reference model from huggingface
    model_ref = hf_model_class.from_pretrained(model_name_or_path, **constructor_kwargs)
    lm_head_parameters = model_ref.lm_head.weight
    del model_ref
    torch.cuda.empty_cache()

    # load the Adapters version of the Llama model
    model = adapters_model_class.from_pretrained(model_name_or_path, **constructor_kwargs)

    if hasattr(model, "add_seq2seq_lm_head"):
        model.add_seq2seq_lm_head("lm_head")
    else:
        model.add_causal_lm_head("lm_head")

    model.heads.lm_head[0].weight = lm_head_parameters

    return model


def get_model(args, device):
    if "t5" in args.model_name_or_path and args.source_prefix is None and args.task_type == "summarization":
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # model
    model_class = AutoModelForSeq2SeqLM
    adapters_model_class = T5AdapterModel
    if "llama" in args.model_name_or_path.lower():
        logger.info("Using LLaMA model")
        model_class = AutoModelForCausalLM
        adapters_model_class = LlamaAdapterModel

    # add peft modules
    if not args.adapter_config_string in ["bitfit", "ln_tuning", "attn_tuning", "full_tuning"]:
        model = load_model_with_adapters_and_lm_head(
            hf_model_class=model_class,
            adapters_model_class=adapters_model_class,
            model_name_or_path=args.model_name_or_path,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            torch_dtype=args.torch_dtype,
        )

        # freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # add PEFT
        if adapters_model_class is T5AdapterModel:
            # T5-3B and T5-11B have a weird setup where key_size != hidden_size // num_heads
            args.adapter_config_string = preprocess_adapter_config_string_for_t5(
                adapter_config_string=args.adapter_config_string,
                model_config=model.config,
            )

        model.add_adapter("adapter", config=args.adapter_config_string, set_active=True)
        model.train()
        model.train_adapter("adapter")
        for name, module in model.named_modules():
            if "adapter" in name:
                module.to(device)
                for param in module.parameters():
                    param.requires_grad = True

    elif args.adapter_config_string == "bitfit":
        model = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.torch_dtype,
        )

        # freeze all but bias parameters
        for name, param in model.named_parameters():
            if not "bias" in name:
                param.requires_grad = False

    elif args.adapter_config_string == "ln_tuning":
        model = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.torch_dtype,
        )

        # freeze all but LN parameters
        for name, param in model.named_parameters():
            # LlaMa layer norm key: input_layernorm, post_attention_layernorm
            # T5 layer norm key:    layer_norm
            if not (("_layernorm" in name) or ("layer_norm" in name)):
                param.requires_grad = False

    elif args.adapter_config_string == "attn_tuning":
        NotImplementedError("Attention tuning is not implmented yet")

    elif args.adapter_config_string == "full_tuning":
        if args.load_in_8bit or args.load_in_4bit:
            logger.error(f"Full tuning can't be applied to quantized models")

        is_llama = "llama" in args.model_name_or_path.lower()
        model_class = AutoModelForCausalLM if is_llama else AutoModelForSeq2SeqLM

        model = model_class.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            **({"use_flash_attention_2": True} if is_llama else {}),
        )

    # send to device if not quantized
    if (not args.load_in_8bit) and (not args.load_in_4bit):
        model = model.to(dtype=args.torch_dtype)

    # adapter is not yet on the device
    if args.load_in_8bit or args.load_in_4bit:
        for name, module in model.named_modules():
            if "adapter" in name:
                module.to(device=device, dtype=args.torch_dtype)

    return model


def get_model_hf(args, device):
    if "t5" in args.model_name_or_path and args.source_prefix is None and args.task_type == "summarization":
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    model_class = AutoModelForSeq2SeqLM
    task_type = peft.TaskType.SEQ_2_SEQ_LM
    is_llama = "llama" in args.model_name_or_path.lower()
    if is_llama:
        logger.info("Using LLaMA model")
        model_class = AutoModelForCausalLM
        task_type = peft.TaskType.CAUSAL_LM

    model = model_class.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        **({"use_flash_attention_2": True} if is_llama else {}),
    )

    method, method_kwargs = peft_comparison.utils.parse_config_string(args.adapter_config_string)
    if method not in peft_comparison.mappings.hf_adapter_config_string_to_peft_args:
        raise ValueError(f"Unknown method {method}, config string={args.adapter_config_string}")

    default_kwargs = peft_comparison.mappings.hf_adapter_config_string_to_peft_args[method]
    method_kwargs = default_kwargs | method_kwargs

    if method in ["hf_lora", "hf_lora_all"]:
        peft_config = peft.LoraConfig(
            task_type=task_type,
            inference_mode=False,
            **method_kwargs,
        )
    elif method == "hf_krona":
        peft_config = peft.LoKrConfig(
            task_type=task_type,
            **method_kwargs,
        )
    elif method == "hf_loha":
        peft_config = peft.LoHaConfig(
            task_type=task_type,
            **method_kwargs,
        )
    else:
        raise ValueError(f"Can't find PEFT config object for method {method}, config string={args.adapter_config_string}")

    model = peft.get_peft_model(model, peft_config)
    model = model.to(device=device, dtype=args.torch_dtype)
    return model


@torch.no_grad()
def evaluate_model(
    model,
    *,
    metric,
    tokenizer,
    dataloader,
    accelerator,
    postprocess_fn,
    target_length,
    max_length=None,
    num_beams=None,
    max_iters=None,
    decoder_only=False,
):
    """
    Args:
        postprocess_fn: a function that takes a pair of lists of strings (predictions, labels) and returns a pair of
            lists of strings (predictions, labels) after postprocessing.
            For an example, look at `peft_comparison.text2text_utils.postprocess_summarization`
    """
    _time = time.time()
    model.eval()
    pbar = tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
        total=max_iters or len(dataloader),
        ncols=80,
    )
    peak_memory_usage = 0
    current_memory_usage = 0
    batch_start_time = time.time()
    throughput_tokens = 0
    throughput_examples = 0
    throughput_batches = 0
    first_10_predictions = []
    first_10_labels = []

    synced_gpus = None
    _ds_plugin = accelerator.deepspeed_plugin
    if _ds_plugin is not None and _ds_plugin.stage == 3:
        synced_gpus = True
        logger.info("Running in stage 3, will use synced gpus for generation")

    for eval_step, batch in enumerate(dataloader):
        pbar.update()
        if max_iters is not None and eval_step > max_iters:
            logger.info(f"{max_iters} evaluation steps reached. Stopping evaluation.")
            break

        unwrapped_model = accelerator.unwrap_model(model)
        generated_tokens = unwrapped_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=max_length,
            max_new_tokens=target_length,
            num_beams=num_beams,
            do_sample=False,
            synced_gpus=synced_gpus,  # stage 3 requires synced gpus or generation may stall
        )

        update_time = time.time() - batch_start_time
        batch_start_time = time.time()

        # track memory usage and throughput
        peak_memory_usage += torch.cuda.max_memory_allocated() / (1024 ** 2)
        current_memory_usage += torch.cuda.memory_allocated() / (1024 ** 2)
        throughput_tokens += batch["input_ids"].numel() / update_time
        throughput_examples += batch["input_ids"].shape[0] / update_time
        throughput_batches += 1 / update_time

        if decoder_only:
            # replace token_ids corresponding to the input text (without the label text i.e. class label name or summary)
            generated_tokens = peft_comparison.text2text_utils.strip_input_tokens_from_generation(
                generated_tokens=generated_tokens,
                len_input_wo_class=[i["input_len"] for i in batch["metadata"]],
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        ).cpu().numpy()

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        labels_str = [_l["targets"] for _l in batch["metadata"]]
        if len(decoded_preds) != len(labels_str):
            print(f"Input ids shape: {batch['input_ids'].shape}"),
            print(f"{len(decoded_preds)} != {len(labels_str)}")

        decoded_preds, labels_str = postprocess_fn(decoded_preds, labels_str)
        metric.add_batch(predictions=decoded_preds, references=labels_str)

        if len(first_10_predictions) < 10:
            first_10_predictions.extend(decoded_preds[:10])
            first_10_labels.extend(labels_str[:10])

        if eval_step == 0:
            logger.info("\n")
            logger.info(f"Example of predictions: {decoded_preds[0]}")
            logger.info(f"Example of labels: {labels_str[0]}")

    if metric.name == "rouge":
        result = metric.compute()#(use_stemmer=True)
    else:
        result = metric.compute()

    # save performance and other metrics to track memory consumption and throughput
    result = {f"eval/{k}": round(v * 100, 4) for k, v in result.items()}
    result["eval/peak_memory_usage"] = peak_memory_usage / (eval_step + 1)
    result["eval/current_memory_usage"] = current_memory_usage / (eval_step + 1)
    result["eval/throughput_tokens"] = throughput_tokens / (eval_step + 1)
    result["eval/throughput_examples"] = throughput_examples / (eval_step + 1)
    result["eval/throughput_batches"] = throughput_batches / (eval_step + 1)

    logger.info("Logging predictions to wandb")
    if wandb.run is not None:
        table = wandb.Table(columns=["Targets", "Predictions"])
        for t, p in zip(first_10_labels, first_10_predictions):
            table.add_data(t, p)
        result["predicitons"] = table

    model.train()
    logger.info(f"Evaluation took {time.time() - _time:.2f} seconds")
    return result


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator(project_dir=args.output_dir, log_with="wandb")
    if not accelerator.is_main_process:
        logger.remove()

    # It's important to get ths after the accelerator initialization
    device = torch.cuda.current_device()
    args.device = device

    if args.total_batch_size is not None:
        if args.gradient_accumulation_steps is not None:
            logger.warning("`--total_batch_size` overrides --gradient_accumulation_steps")
        if args.total_batch_size % (accelerator.num_processes * args.per_device_train_batch_size) != 0:
            raise ValueError(f"`--total_batch_size` ({args.total_batch_size}) is not divisible by "
                             f"num_processes * per_device_train_batch_size ({accelerator.num_processes} * {args.per_device_train_batch_size})")
        logger.info(f"accelerator.num_processes: {accelerator.num_processes}")
        args.gradient_accumulation_steps = args.total_batch_size // (args.per_device_train_batch_size * accelerator.num_processes)
        logger.info(f"Setting gradient accumulation steps to {args.gradient_accumulation_steps}.")
    else:
        args.total_batch_size = args.gradient_accumulation_steps * args.per_device_train_batch_size * accelerator.num_processes

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    accelerator.init_trackers(
        args.wandb_project,
        init_kwargs={"wandb": {
            "tags": args.tags,
            "entity": "text_machine_lab_babylm",
        }})

    if accelerator.is_main_process:
        wandb.save(os.path.abspath(__file__), policy="now") # save current script

    logger.info("*" * 40)
    for k, v in vars(args).items():
        logger.info(f"{k:30}: {v}")
    logger.info("*" * 40)

    # Load pretrained model and tokenizer
    use_adapters_library = not args.adapter_config_string.startswith("hf_")
    if use_adapters_library:
        model = get_model(args, device=device)
    else:
        model = get_model_hf(args, device=device)
    torch.cuda.reset_peak_memory_stats()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=max(args.max_source_length, args.max_target_length) + 2,  # void the annoying warning messge, +2 is for bos and eos
    )

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.config.pad_token_id = model.config.eos_token_id

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        logger.warning(f"Tokenizer vocab size ({len(tokenizer)}) does not match the model "
                       f"embedding size ({embedding_size}).")
        if len(tokenizer) > embedding_size:
            raise ValueError("Tokenizer vocab size is larger than the model embedding size. "
                             "Probably you're using a wrong tokenizer/model combination. ")

    logger.info(model)

    dtype_counts = {}
    for p in model.parameters():
        dtype_counts[str(p.dtype)] = dtype_counts.get(str(p.dtype), 0) + p.numel()

    if len(dtype_counts) == 0:
        logger.warning("No parameters in the model?")

    logger.info(dtype_counts)

    total_parameters = sum(dtype_counts.values())
    dtype_info = [f"{dtype}: {count} ({count / total_parameters * 100:.2f}%)" for dtype, count in dtype_counts.items()]
    logger.info("Model dtypes: ", " | ".join(dtype_info))

    # we need to add padding token to llama
    if args.decoder_only:
        tokenizer.add_special_tokens({'pad_token': tokenizer.bos_token})

    ############################################
    # Data preprocessing
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if args.subsample_data is not None:
        logger.warning(f"Subsampling the dataset to {args.subsample_data} first examples.")
        # remember that it's dataset dict
        def get_subsample_data(subset_name):
            if subset_name == "train": return args.subsample_data
            return max(100, args.subsample_data // 10)
        raw_datasets = {k: v.select(range(get_subsample_data(k))) for k, v in raw_datasets.items()}

    _dataset_name_for_preprocessing = args.dataset_name
    if args.task_type == "classification":
        _dataset_name_for_preprocessing = args.dataset_config_name

    if args.dataset_name in ["cnn_dailymail", "EdinburghNLP/xsum"]:
        # 1600 is selected based on the MAM paper
        # sample 1600 examples for validation and test deterministically
        logger.warning(f"Subsampling the dataset to 1600 first examples for validation and test.")
        raw_datasets["validation"] = raw_datasets["validation"].select(range(1600))
        raw_datasets["test"] = raw_datasets["test"].select(range(1600))

    raw_datasets, postprocess_fn = peft_comparison.text2text_utils.dataset_to_text2text(
        raw_datasets,
        task_type=args.task_type,
        dataset_name=_dataset_name_for_preprocessing,
        decoder_only=args.decoder_only,
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    padding = "max_length" if args.pad_to_max_length else False
    def preprocess_function(examples, is_eval=False, decoder_only=False):
        inputs = examples["source_text"]
        targets = examples["target_text"]
        inputs = [args.source_prefix + inp for inp in inputs]

        if not decoder_only:
            # T5
            model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
            labels = tokenizer(text_target=targets, max_length=args.max_target_length, padding=padding, truncation=True)
            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
            if is_eval:
                model_inputs["metadata"] = [{"targets": t} for t in targets]

        else:
            # @NOTE: the way we have written preprocessing and collation for llama,
            # - set padding=False in preprocessing (so that we know what's the max_len in the batch)
            # - set padding=True in collation (so that we can pad to the multiple of 8 > max_len in the batch)
            # - we can set labels to input_ids because the token shifting is taken care of in the modeling_llaama file
            if is_eval:
                model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=False, truncation=True)
            else:
                inputs = [i + " " + t for i, t in zip(inputs, targets)]
                model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=False, truncation=True)
            model_inputs["labels"] = model_inputs["input_ids"]
            if is_eval:
                input_wo_label = tokenizer(inputs, max_length=args.max_source_length, padding=False, truncation=False)
                input_wo_label = input_wo_label["input_ids"]
                model_inputs["metadata"] = []
                for idx in range(len(targets)):
                    model_inputs["metadata"].append({
                        "targets": targets[idx],
                        "input_len": len(input_wo_label[idx]),
                    })

        return model_inputs

    with accelerator.main_process_first():
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on val dataset  ",
            fn_kwargs={"is_eval": True, "decoder_only": args.decoder_only},
        )
        test_dataset = eval_dataset
        if "test" in raw_datasets:
            test_dataset = raw_datasets["test"].map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                fn_kwargs={"is_eval": True, "decoder_only": args.decoder_only},
            )

        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            batch_size=min(5000, len(raw_datasets["train"]) // args.preprocessing_num_workers),
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
            fn_kwargs={"decoder_only": args.decoder_only}
        )

    if len(raw_datasets["validation"]) < 1_000:
        logger.warning(f"Validation dataset is small ({raw_datasets['validation']}), running full validation set during training.")
        args.max_eval_steps_durig_validation = None

    # Log a few random samples from the training set:
    if args.verbosity > 1:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set:")
            for k, v in train_dataset[index].items():
                if hasattr(v, "shape"):
                    logger.info(f"  {k}.shape: {v.shape}")
                logger.info(f"  {k}: {v}")

    label_pad_token_id = -100
    if not args.decoder_only:
        data_collator = DataCollatorForSeq2SeqWithMetadata(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8,
        )
    else:
        data_collator = DataCollatorForCausalLMWithMetadata(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_source_length,
            pad_to_multiple_of=8,
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = eval_dataloader
    if "test" in raw_datasets:
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm", "layer_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.total_parameters = total_parameters
    args.trainable_parameters = trainable_parameters
    logger.info(f"Total model parameters: {total_parameters:,}")
    logger.info(f"Trainable parameters  : {trainable_parameters:,} ({trainable_parameters / total_parameters * 100:.4f}%)")
    if args.verbosity > 1:
        logger.info("Trainable model parameters")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name}: {param.numel():,}")
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.max_train_steps < args.min_train_steps:
        args.max_train_steps = args.min_train_steps
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        overrode_max_train_steps = True
        logger.warning(f"Overriding `max_train_steps` to {args.min_train_steps} "
                       f"and `args.num_train_epochs` to {args.num_train_epochs} "
                       f"to meet `min_train_steps` requirement.")

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    if args.eval_every_steps is None:
        args.eval_every_steps = min(2000, args.max_train_steps // 10)
        logger.info(f"Setting `eval_every_steps` to {args.eval_every_steps}.")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_train_steps * args.lr_scheduler_warmup_percent),
        num_training_steps=args.max_train_steps,
    )

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    if accelerator.is_main_process:
        wandb.config.update(experiment_config)

    # Metric
    if args.task_type == "summarization":
        metric = evaluate.load("rouge")
    else:
        metric = evaluate.load("super_glue", args.dataset_config_name)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total warmup steps for LR = {int(args.max_train_steps * args.lr_scheduler_warmup_percent)}")

    # Only show the progress bar once on each machine.
    progress_bar = trange(
        args.max_train_steps,
        desc="Training (total steps)",
        disable=not accelerator.is_local_main_process,
        total=args.max_train_steps,
        ncols=80,
        leave=True,
    )

    update_step = 0
    starting_epoch = 0
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            update_step = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            update_step = resume_step // args.gradient_accumulation_stepp

        progress_bar.update(update_step)

    active_dataloader = train_dataloader
    if args.resume_from_checkpoint and resume_step is not None:
        active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)

    tokens_seen = 0
    tokens_seen_before = 0
    batches_in_update = args.gradient_accumulation_steps * accelerator.num_processes
    main_metric_name = "eval/accuracy" if args.task_type == "classification" else "eval/rougeL"
    best_metric_value = 0
    best_results_dict = None

    prof = None
    if args.profile:
        prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
                record_shapes=True,
                with_stack=True)

    for epoch in range(starting_epoch, args.num_train_epochs):
        logger.info(f"Starting epoch {epoch + 1} / {args.num_train_epochs}")
        model.train()

        if prof is not None: prof.start()
        for batch_idx, batch in enumerate(active_dataloader):
            if prof is not None: prof.step()
            # vars for calculating throughput
            batch_start_time = time.time()
            tokens_seen += batch["input_ids"].numel() * accelerator.num_processes

            # print first batch
            if batch_idx == 0 and epoch == 0:
                logger.info("============= CHECKING FIRST BATCH =============")
                logger.info("Tensor shapes: ")
                logger.info(batch["input_ids"].shape)
                logger.info("Decoded text of first example in the batch:")
                s_text = tokenizer.decode(batch["input_ids"][0, :], skip_special_tokens=False)
                logger.info(f"input_ids text: {s_text}")
                if "decoder_input_ids" in batch:
                    t_text = tokenizer.decode(batch["decoder_input_ids"][0, :], skip_special_tokens=False)
                    logger.info(f"decoder_input_ids text: {t_text}")

                t_ids = batch["labels"][0, :]
                t_ids[t_ids == -100] = tokenizer.pad_token_id
                t_text = tokenizer.decode(t_ids, skip_special_tokens=False)
                logger.info(f"Labels text: {t_text}")

            global_step += 1
            outputs = model(**batch)
            loss = outputs.loss
            loss_for_backward = loss / args.gradient_accumulation_steps
            accelerator.backward(loss_for_backward)

            if not(global_step % args.gradient_accumulation_steps == 0 or global_step == len(train_dataloader) - 1):
                continue
            # the code below is only executed on the update step

            progress_bar.update()
            optimizer.step()
            lr_scheduler.step()

            # log memory and thoughput
            if (update_step % 10) == 0:
                tokens_in_update = tokens_seen - tokens_seen_before
                peak_memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in megabytes
                current_memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)
                update_time = time.time() - batch_start_time
                batch_start_time = time.time()
                accelerator.log(
                    {
                        "throughput_tokens": tokens_in_update / update_time,
                        "throughput_examples": args.total_batch_size / update_time,
                        "throughput_batches": batches_in_update / update_time,
                        "peak_memory_usage": peak_memory_usage,
                        "current_memory_usage": current_memory_usage,
                    },
                    step=update_step,
                )
                tokens_seen_before = tokens_seen

            # clear grads
            optimizer.zero_grad()
            update_step += 1

            if update_step >= args.max_train_steps:
                logger.info("Max number of steps reached. Stopping training")
                break

            accelerator.log(
                {
                    "train/loss": loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "update_step": update_step,
                    "global_steps": global_step,
                },
                step=update_step,
            )

            if (update_step + 1) % args.eval_every_steps == 0 or update_step == 1:
                logger.info(f"Evaluating model at, update step: {update_step}, global step: {global_step}")
                result = evaluate_model(
                    model=model,
                    metric=metric,
                    tokenizer=tokenizer,
                    dataloader=eval_dataloader,
                    accelerator=accelerator,
                    max_length=(args.max_source_length + args.max_target_length) if args.decoder_only else args.val_max_target_length,
                    target_length=args.val_max_target_length,
                    num_beams=1,
                    max_iters=args.max_eval_steps_durig_validation,
                    postprocess_fn=postprocess_fn,
                    decoder_only=args.decoder_only
                )
                accelerator.log(result, step=update_step)
                if result[main_metric_name] > best_metric_value:
                    logger.info(f"New best metric found: {result[main_metric_name]} at update step {update_step}")
                    best_metric_value = result[main_metric_name]
                    best_results_dict = result.copy()

    # final evaluation
    # @NOTE: we want to do atleast one evaluation with beam search
    update_step += 1
    logger.info(f"Final evaluation, validation set, {update_step=}")

    result = evaluate_model(
        model=model,
        metric=metric,
        tokenizer=tokenizer,
        dataloader=eval_dataloader,
        accelerator=accelerator,
        max_length=(args.max_source_length + args.max_target_length) if args.decoder_only else args.val_max_target_length,
        target_length=args.val_max_target_length,
        num_beams=args.num_beams,
        max_iters=None,
        postprocess_fn=postprocess_fn,
        decoder_only=args.decoder_only
    )
    if result[main_metric_name] > best_metric_value:
        logger.info(f"New best metric found: {result[main_metric_name]} at update step {update_step}")
        best_metric_value = result[main_metric_name]
        best_results_dict = result.copy()

    logger.info(pformat(result))
    accelerator.log(result, step=update_step)

    accelerator.log(best_results_dict, step=update_step)

    # Eval on test set
    if "test" in raw_datasets:
        logger.info(f"Final evaluation, test set, {update_step=}")
        result = evaluate_model(
            model=model,
            metric=metric,
            tokenizer=tokenizer,
            dataloader=test_dataloader,
            accelerator=accelerator,
            max_length=(args.max_source_length + args.max_target_length) if args.decoder_only else args.val_max_target_length,
            target_length=args.val_max_target_length,
            num_beams=args.num_beams,
            max_iters=None,
            postprocess_fn=postprocess_fn,
            decoder_only=args.decoder_only
        )
        result = {f"test/{k.lstrip('eval/')}": v for k, v in result.items()}
        accelerator.log(result, step=update_step)

    # save results and all arguments
    all_results = best_results_dict.copy()
    all_results["args"] = vars(args)
    for k, v in all_results["args"].items():
        if not isinstance(v, (float, int, bool, str, list)):
            all_results["args"][k] = str(v)

    print(f"Saving results to {args.output_dir}")
    if os.path.exists(args.output_dir):
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            _all_results = {k: v for k, v in all_results.items() if not isinstance(v, wandb.Table)}
            json.dump(_all_results, f, indent=4)

    logger.info("Script successfully finished!")


if __name__ == "__main__":
    main()
