# -*- coding: utf-8 -*-
from typing import Optional
from accelerate import PartialState
from transformers import PreTrainedTokenizer
import argparse
from dataclasses import dataclass, field
from datasets import Dataset

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
from rner import InputExample
from rner.trainer_utils import get_model, get_tokenizer
from rner.prompt import TEMPLATE_MAP


@dataclass
class RNERConfig:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset path."}
    )
    data_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset Cache path."}
    )
    add_source: bool = field(
        default=False,
        metadata={
            "help": "Whether to add a comment in the prompt indicating which dataset the sample comes from."
        }
    )
    add_cot: bool = field(
        default=False,
        metadata={
            "help": "Whether to use chain‑of‑thought training"
        }
    )
    template: Optional[str] = field(
        default="qwen",
        metadata={
            "help": "The template to use for the prompt."
        }
    )
    preprocess_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of workers to use for preprocessing."
        }
    )


def process_example(
        example,
        add_source: bool = False,
        add_cot: bool = False
):
    input_example = InputExample.from_dict(example)
    return {
        "user": input_example.to_prompt(add_source=add_source),
        "assistant": input_example.to_completion(add_cot=add_cot)
    }


def tokenize(
        examples,
        template: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 8192
):
    chat_template = TEMPLATE_MAP[template]

    start_token_id = None
    if "start_token" in chat_template and chat_template['start_token'] is not None:
        start_token_id = tokenizer.convert_tokens_to_ids([chat_template['start_token']])[0]

    all_user, all_assistant = [], []
    for user, assistant in zip(examples["user"], examples["assistant"]):
        user_msg = chat_template['user'].replace("{{content}}", user)
        assistant_msg = chat_template['assistant'].replace("{{content}}", assistant)
        all_user.append(user_msg)
        all_assistant.append(assistant_msg)

    user_input_ids = tokenizer(
        all_user,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )['input_ids']

    assistant_input_ids = tokenizer(
        all_assistant,
        add_special_tokens=False,
        padding=False,
        truncation=False
    )['input_ids']

    all_input_ids = []
    all_labels = []
    for user_input_id, assistant_input_id in zip(user_input_ids, assistant_input_ids):
        input_ids = user_input_id + assistant_input_id
        labels = [-100] * len(user_input_id) + assistant_input_id

        if start_token_id is not None:
            input_ids = [start_token_id] + input_ids
            labels = [-100] + labels
        if tokenizer.eos_token_id is not None and input_ids[-1] != tokenizer.eos_token_id:
            input_ids = input_ids + [tokenizer.eos_token_id]
            labels = labels + [tokenizer.eos_token_id]

        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        if all([l == -100 for l in labels]):
            continue

        all_input_ids.append(input_ids)
        all_labels.append(labels)
    return {
        "input_ids": all_input_ids,
        "labels": all_labels
    }


def main(rner_config: RNERConfig, train_config: SFTConfig, model_config: ModelConfig):
    model = get_model(model_config, train_config)

    tokenizer = get_tokenizer(model_config)

    dataset = Dataset.from_json(rner_config.data_path, cache_dir=rner_config.data_cache_path)
    with PartialState().local_main_process_first():
        dataset = dataset.map(
            process_example,
            batched=False,
            fn_kwargs={
                "add_source": rner_config.add_source,
                "add_cot": rner_config.add_cot
            },
            desc="Formating Examples",
            remove_columns=list(dataset.column_names)
        )
        dataset = dataset.map(
            tokenize,
            batched=True,
            fn_kwargs={
                "template": rner_config.template,
                "tokenizer": tokenizer,
                "max_length": train_config.max_length
            },
            num_proc=rner_config.preprocess_workers,
            desc="Tokenizing examples",
            remove_columns=list(dataset.column_names)
        )
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(train_config.output_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (RNERConfig, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    rner_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(rner_args, training_args, model_args)
