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
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import numpy as np

import nltk
import datasets
import evaluate
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Optional, Union
from dataclasses import dataclass, field
import wandb

import transformers
from peft import (
    PromptTuningConfig,
    PrefixTuningConfig,
    LoraConfig, 
    IA3Config,
    get_peft_model,
    PromptTuningInit, 
    PromptTuningConfig, 
    TaskType, 
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    BitsAndBytesConfig,
    set_seed,
    DataCollatorForSeq2Seq,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.34.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`

nltk.data.find("tokenizers/punkt")

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

seq2seq_models = set([
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
])

def define_peft_config(args):    

    if args.peft_method in ["lora"]:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if args.model_name_or_path in seq2seq_models else "CAUSAL_LM",
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
        )

    elif args.peft_method in ["adapters"]:
        peft_config = None

    elif args.peft_method in ["ia_3"]:
        peft_config = IA3Config(
            task_type=TaskType.SEQ_2_SEQ_LM if args.model_name_or_path in seq2seq_models else "CAUSAL_LM",
            target_modules=args.target_modules,
            feedforward_modules=args.feedforward_modules,
            init_ia3_weights=args.init_ia3_weights,
        )

    elif args.peft_method in ["prompt_tuning", "p_tuning"]:
        print(args.num_virtual_tokens)
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if args.model_name_or_path in seq2seq_models else "CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.TEXT if args.prompt_tuning_init=="text" else PromptTuningInit.RANDOM,
            prompt_tuning_init_text=args.prompt_tuning_init_text if args.prompt_tuning_init=="text" else None,
            num_virtual_tokens=args.num_virtual_tokens,
            tokenizer_name_or_path=args.model_name_or_path,
        )
    
    elif args.peft_method in ["prefix_tuning"]:
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if args.model_name_or_path in seq2seq_models else "CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            prefix_projection=args.prefix_projection,
        )


    return peft_config


def get_model(args):
    # check free space
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    # how to load the model
    if args.load_in_4bit:
        args.load_in_8bit = False

    # define model
    if args.use_quantization: 
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                llm_int8_threshold=args.llm_int8_threshold,
                bnb_4bit_compute_dtype=args.torch_dtype,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            ),
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(args.torch_dtype)

    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is not None:
        print(f"Using present PAD token in the tokenizer: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    else:
        set_pad_to = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': set_pad_to})
        model.config.pad_token_id = model.config.eos_token_id
        print(f"Pointing PAD token to: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

    # patch the model with PEFT config
    peft_config = define_peft_config(args)
    model = get_peft_model(model, peft_config)
    print(model)

    # Verifying the datatypes.
    print("\nPrecision details:")
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    # verying the size of the model
    par_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    par_fixed = sum(p.numel() for p in model.parameters())
    par_percent = int(100 * par_trainable / par_fixed)
    print(f"\nTotal number of trainable parameters: {par_trainable:,} ({par_percent}%)")

    return model, tokenizer, model.config


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cnn_dailymail",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="3.0.0",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="t5-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_warmup_percent", type=float, default=0.06, help="Percentage of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=None,
        help="Evaluate model after these many steps.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        #choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    # manually adding arguments for quantization and for LoRA module
    parser.add_argument(
        "--use_quantization", 
        type=bool, 
        default=False, 
        help=("enable 4 or 8bit quantization."),
    )
    parser.add_argument(
        "--device_map", 
        type=str, 
        default=None,#"auto"
        help=("Which GPU/s to use for hosting the model"),
    )
    
    parser.add_argument(
        "--load_in_8bit", 
        type=bool, 
        default=False, 
        help=("enable 8bit quantization."),
    )
    parser.add_argument(
        "--llm_int8_threshold",
        type=float,
        default=6.0, 
        help=("value of the outliner threshold. only relevant when load_in_8bit=True"),
    )

    parser.add_argument(
        "--load_in_4bit",
        type=bool,
        default=True, 
        help=("enable 4bit quantization."),
    )

    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="fp4",
        help=("set the quantization data type in the `bnb.nn.Linear4Bit` layers. Options are {'fp4','np4'}."),
    )

    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        type=bool,
        default=False,
        help=("enable nested quantization where the quantization constants from the first quantization are quantized again."),
    )

    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=bool,
        default="fp16",
        help=(
            "This sets the computational type which might be different than the input time. For example, inputs might be "
            "fp32, but computation can be set to bf16 for speedups. Options are {'fp32','fp16','bf16'}."
        ),
    )

    parser.add_argument(
        "--torch_dtype", 
        type=torch.dtype,
        default=torch.bfloat16,
        help=(
            "this sets the dtype of the remaining non quantized layers. `bitsandbytes` library suggests to set the value"
            "to `torch.float16` for 8 bit model and use the same dtype as the compute dtype for 4 bit model "
        ),
    )

    parser.add_argument(
        "--skip_modules",
        type=List[str],
        default=None,
        help=(
            "an explicit list of the modules that we don't quantize. The dtype of these modules will be `torch_dtype`."
        ),
    )

    parser.add_argument(
        "--keep_in_fp32_modules", 
        type=List[str],
        default=None,
        help=("an explicit list of the modules that we don't quantize. We keep them in `torch.float32`."),
    )


    # lora arguments
    parser.add_argument(
        "--peft_method",
        type=str,
        default=None, 
        choices=["lora", "prompt_tuning", "p_tuning", "prefix_tuning", "ia_3", "adapters"],
        help=("Lora attention dimension"),
    )
    
    parser.add_argument(
        "--r",
        type=int,
        default=8, 
        help=("Lora attention dimension"),
    )

    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,#["q_proj", "v_proj"],#['k', 'v'],#['q', 'v'],
        help=(
            "List of module names or regex expression of the module names to replace with Lora."
            "For example, 'q,v' or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        ),
    )

    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16, 
        help=("Lora alpha"),
    )

    parser.add_argument(
        "--lora_dropout",
        type=float, 
        default=0.0, 
        help=("Lora dropout"),
    )

    parser.add_argument(
        "--fan_in_fan_out",
        type=bool,
        default=True,
        help=("Set this to True if the layer to replace stores weight like (fan_in, fan_out)"),
    )

    parser.add_argument(
        "--bias", 
        type=str, 
        default="none", 
        help=("Bias type for Lora. Can be 'none', 'all' or 'lora_only'"),
    )

    parser.add_argument(
        "--modules_to_save", 
        type=Optional[List[str]],
        default=None,
        help=(
            "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        ),
    )

    parser.add_argument(
        "--init_lora_weights", 
        type=bool,
        default=True,
        help=(
            "Whether to initialize the weights of the Lora layers with their default initialization. Don't change "
            "this setting, except if you know exactly what you're doing."
        ),

    )

    parser.add_argument(
        "--layers_to_transform", 
        type=Optional[Union[List[int], int]],
        default=None,
        help=(
            "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        ),
    )

    parser.add_argument(
        "--layers_pattern", 
        type=Optional[str],
        default=None,
        help=(
            "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        ),
    )

    # promp-tuning, prefix-tuning and p-tuning arguments
    parser.add_argument(
        "--num_virtual_tokens", 
        type=int,
        default=20,
        help=("how many virtual tokens to add to vocabulary. Embeddings for these tokens will be tuned in the fine-tuning process."),
    )
    parser.add_argument(
        "--prompt_tuning_init", 
        type=str,
        default="text",
        help=("Initialize virtual tokens from text or randomly"),
    )
    parser.add_argument(
        "--prompt_tuning_init_text", 
        type=str,
        default="text",
        help=("If initialization strategy is \"Text\" then, text given to this argument will be used to initialize the virtual tokens"),
    )
    parser.add_argument(
        "--prefix_projection", 
        type=bool,
        default=True,
        help=("Use a two-layer MLP to encode the prefix"),
    )
    

    # (IA)3 arguments (some arguments are same as LoRA, so not added here)
    parser.add_argument(
        "--feedforward_modules", 
        type=Optional[Union[List[str], str]],
        default=None,
        help=(
            "List of module names or a regex expression of module names which are feedforward" 
            "For example, ['output.dense']"
        ),
    )
    parser.add_argument(
        "--init_ia3_weights", 
        type=bool,
        default=True,
        help=("Whether to initialize the vectors in the (IA)^3 layers."),
    )

    # wandb arguments
    parser.add_argument(
        "--wandb_project", 
        type=str,
        default="PEFT_comparison",
        help=("name to be given to Weight and Biases logging repository"),
    )

    parser.add_argument(
        "--wandb_tags", 
        type=list,
        default=["trial", "t5-base", "classification"],
        help=("tags to be given to individual runs in WandB repository"),
    )

    parser.add_argument(
        "--wandb_name", 
        type=str,
        default=None,
        help=("display name for the run"),
    )

    
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.target_modules:
        if "," in args.target_modules:
            args.target_modules = args.target_modules.split(",")
        if "*" in args.target_modules and "," in args.target_modules:
            raise NotImplementedError("Combining * and , in target_modules is not supported yet.")
        

    return args


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_summarization_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    #accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    accelerator = Accelerator(**accelerator_log_kwargs)
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model, tokenizer, config = get_model(args)
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Temporarily set max_target_length for validation.
        max_target_length = args.val_max_target_length
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on val dataset",
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_train_steps * args.lr_scheduler_warmup_percent),#args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    # Metric
    metric = evaluate.load("rouge")

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
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
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

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_stepp

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # start wandb logging
    wandb.init(
        project=args.wandb_project,
        config=args,
        #tags=[args.model_name_or_path, args.task_name, args.peft_method],
    )

    global_steps = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            global_steps += 1
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            # to wandb
            wandb.log(
                {
                    "Train/loss_per_effective_batch": loss,
                    "Train/learning_rate": optimizer.param_groups[0]["lr"],
                    "Train/epoch": epoch,
                    "Train/par_updates": completed_steps,
                    "Train/global_steps": global_steps,
                },
                step=completed_steps,
            )
            
            if (step) % args.eval_every_steps == 0 or step == len(train_dataloader) - 1:
                model.eval()
                gen_kwargs = {
                    "max_length": args.val_max_target_length,
                    "num_beams": args.num_beams,
                }
                for step_eval, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        unwrapped_model = accelerator.unwrap_model(model)
                        generated_tokens = unwrapped_model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )

                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]
                        if not args.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                        generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                        generated_tokens = generated_tokens.cpu().numpy()
                        labels = labels.cpu().numpy()

                        if args.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                        metric.add_batch(
                            predictions=decoded_preds,
                            references=decoded_labels,
                        )
                    
                    #
                    if (not step == len(train_dataloader) - 1) and (step_eval > 10):
                        break

                result = metric.compute(use_stemmer=True)
                result = {k: round(v * 100, 4) for k, v in result.items()}

                logger.info(result)

                if args.with_tracking:
                    result["train_loss"] = total_loss.item() / len(train_dataloader)
                    result["epoch"] = epoch
                    result["step"] = completed_steps
                    accelerator.log(result, step=completed_steps)

                # to wandb
                wandb.log(
                    {
                        "Eval/results": result,
                        "Eval/total_train_loss": total_loss.item() / len(train_dataloader),
                        "Eval/epoch": epoch,
                        "Eval/par_updates": completed_steps,
                        "Eval/global_steps": global_steps,
                    },
                    step=completed_steps,
                )
                model.train()
    
    # save results
    all_results = {f"eval_{k}": v for k, v in result.items()}
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f)
    
    # save all arguments
    with open(os.path.join(args.output_dir, "all_inputs.json"), "w") as f:
        args_dict = vars(args)
        for k, v in args_dict.items():
            if not isinstance(v, (float, int, bool, str, list)):
                args_dict[k] = str(v)
        json.dump(args_dict, f, indent=4)


if __name__ == "__main__":
    main()