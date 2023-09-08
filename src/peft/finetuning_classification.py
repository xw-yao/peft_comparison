# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import sklearn
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import wandb

import transformers
from peft import (
    PromptTuningConfig,
    LoraConfig, 
    IA3Config,
    get_peft_model,
    PromptTuningInit, 
    PromptTuningConfig, 
    TaskType, 
    PeftType,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
    set_seed,
    #AutoAdapterModel,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.33.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("premise", "hypothesis"),#("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("passage", "question"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1"),# "choice2", "question"),
    "multirc": ("paragraph", "question"),#, "answer"),
}

def define_peft_config(args):
    

    if args.peft_method in ["lora"]:
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,#["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=args.lora_dropout,
            bias=args.bias,#"none",
        )

    elif args.peft_method in ["adapters"]:
        peft_config = None

    elif args.peft_method in ["ia_3"]:
        peft_config = IA3Config(
            task_type="SEQ_CLS",
            target_modules=args.target_modules,
            feedforward_modules=args.feedforward_modules,
            init_ia3_weights=args.init_ia3_weights,
        )

    elif args.peft_method in ["prompt_tuning", "p_tuning", "prefix_tuning"]:
        peft_config = PromptTuningConfig(
            task_type="SEQ_CLS",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            #prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
            num_virtual_tokens=args.num_virtual_tokens,
            tokenizer_name_or_path=args.model_name_or_path,
        )


    return peft_config

#
def get_model(args, num_labels):
    
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
        #model = AutoModelForCausalLM.from_pretrained(
        #model = AutoModelForConditionalGeneration.from_pretrained(
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            #max_memory=max_memory,
            num_labels=num_labels,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                llm_int8_threshold=args.llm_int8_threshold,
                #llm_int8_has_fp16_weight=quant_args.llm_int8_has_fp16_weight,
                bnb_4bit_compute_dtype=args.torch_dtype,#torch.float16,#torch.float16,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            ),
            torch_dtype=args.torch_dtype,#torch.float16,#torch.bfloat16,
            device_map=args.device_map,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            #max_memory=max_memory,
            num_labels=num_labels,
            torch_dtype=args.torch_dtype,#torch.float16,#torch.bfloat16,
            device_map=args.device_map,
        )
        
    #print(model)

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(args.torch_dtype)

    # keeping the model output in float-32 for LM-Head
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            #print(x.dtype)
            return super().forward(x).to(args.torch_dtype)
    #model.lm_head = CastOutputToFloat(model.lm_head)
    # @TODO: ask Vlad, do we need this for classification?
    if 't5' in args.model_name_or_path:
        model.classification_head = CastOutputToFloat(model.classification_head)
    else:
        model.score = CastOutputToFloat(model.score)
    
    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is not None:
        print(f"Using present PAD token in the tokenizer: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    else:
        set_pad_to = tokenizer.eos_token
        #set_pad_id = tokenizer.eos_token_id
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
    print(f"Total number of trainable parameters: {par_trainable:,} ({par_percent}%)")


    return model, tokenizer, model.config


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default="cb",
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
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
        default="t5-base",#"meta-llama/Llama-2-7b-hf",#"t5-base",#"facebook/bart-base",#"t5-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        #required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
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
    parser.add_argument(
        "--lr_scheduler_warmup_percent", type=float, default=0.06, help="Percentage of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
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
        default=False,
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
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
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
        default=None,#["classification_head"],
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
        default="lora", 
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
        type=Optional[Union[List[str], str]],
        default=["q_proj", "v_proj"],#["q_proj", "v_proj"],#['k', 'v'],#['q', 'v'],
        help=(
            "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
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

    # promp-tuning arguments
    parser.add_argument(
        "--num_virtual_tokens", 
        type=int,
        default=10,
        help=("how many virtual tokens to add to vocabulary. Embeddings for these tokens will be tuned in the fine-tuning process."),
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

    # parse
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_glue_no_trainer", args)

    # fix the random seed
    set_seed(args.seed)
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = load_dataset("super_glue", args.task_name)
    """
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        #raw_datasets = load_dataset("glue", args.task_name)
        raw_datasets = load_dataset("super_glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    """
    
    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    """
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )
    """
    model, tokenizer, config = get_model(args, num_labels)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        if args.task_name == "copa":
            texts = (
                (examples["premise"], examples["choice1"], examples["choice1"], examples["question"])
            )
        elif args.task_name == "multirc":
            texts = (
                (examples["paragraph"], examples["question"], examples["answer"])
            )
        else:
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
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
        num_warmup_steps=int(args.max_train_steps * args.lr_scheduler_warmup_percent),#max(100, int(args.max_train_steps * args.lr_scheduler_warmup_percent)),#args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
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
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    """
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")
    """
    metric = evaluate.load("super_glue", args.task_name)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
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
            completed_steps = resume_step // args.gradient_accumulation_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    # start wandb logging
    wandb.init(
        project=args.wandb_project,
        config=args,
        #tags=[args.model_name_or_path, args.task_name, args.peft_method],
    )    
    #
    global_steps = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if True: #args.with_tracking:
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
            if True: #args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

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

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        # to wandb
        wandb.log(
            {
                "Eval/accuracy" if args.task_name is not None else "Eval/glue": eval_metric,
                "Eval/total_train_loss": total_loss.item() / len(train_dataloader),
                "Eval/epoch": epoch,
                "Eval/par_updates": completed_steps,
                "Eval/global_steps": global_steps,
            },
            step=completed_steps,
        )

        #
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)

    #
    wandb.save(os.path.abspath(__file__), policy="now") # save current script
    wandb.finish()
    
    return

if __name__ == "__main__":
    main()






