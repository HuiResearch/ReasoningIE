# -*- coding: utf-8 -*-
# Project : ReasoningNER
# Time    : 2025/7/30 15:13
# Author  : Hui Huang
import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from trl import ModelConfig, GRPOConfig, GRPOTrainer, get_peft_config, TrlParser

from rner import InputExample, compute_f1
from rner.trainer_utils import get_model, get_tokenizer
from rner.prompt import build_prompt
from rner.utils import process_completion


@dataclass
class RNERConfig:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset path for GRPO."}
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
    template: Optional[str] = field(
        default="qwen",
        metadata={
            "help": "The template to use for the prompt."
        }
    )


def make_prompt(
        example,
        add_source: bool = False,
        template: str = 'qwen',
):
    input_example = InputExample.from_dict(example)
    prompt = input_example.to_prompt(add_source=add_source)
    prompt = build_prompt(prompt, template)
    return {
        "prompt": prompt,
        "golds": input_example.to_completion(add_cot=False),
        "schema": input_example.schema
    }


def f1_reward(completions, golds, **kwargs):
    rewards = []
    for completion, gold in zip(completions, golds):
        answer = process_completion(completion)
        try:
            score = compute_f1([answer], [gold])
        except:
            score = 0.

        rewards.append(score)
    return rewards


def schema_reward(completions, schema, **kwargs):
    rewards = []
    for completion, labels in zip(completions, schema):
        try:
            answer = process_completion(completion)
            # 可以避免gold无实体、模型也没有预测出结果的情况
            json.loads(answer)
            example = InputExample.from_completion(answer)
            reward = 1
            for entity in example.entities:
                if entity.label not in labels:
                    reward = 0
                    break
        except:
            reward = 0.
        rewards.append(reward)
    return rewards


def main(rner_config: RNERConfig, training_config: GRPOConfig, model_config: ModelConfig):
    dataset = Dataset.from_json(rner_config.data_path, cache_dir=rner_config.data_cache_path)
    dataset = dataset.map(
        make_prompt,
        batched=False,
        fn_kwargs={
            "add_source": rner_config.add_source,
            "template": rner_config.template
        },
        desc="Formating Examples",
        remove_columns=list(dataset.column_names)
    )

    model = get_model(model_config, training_config)

    tokenizer = get_tokenizer(model_config)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[f1_reward, schema_reward],
        args=training_config,
        train_dataset=dataset,
        peft_config=get_peft_config(model_config),
        processing_class=tokenizer
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(training_config.output_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (RNERConfig, GRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    rner_args, training_args, model_args = parser.parse_args_and_config()
    main(rner_args, training_args, model_args)
