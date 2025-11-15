# -*- coding: utf-8 -*-
# Project : ReasoningNER
# Time    : 2025/7/31 10:47
# Author  : Hui Huang
import argparse
import json
import os
from typing import Literal

from datasets import Dataset

from rner import load_data, InputExample
import random


def sample_for_grpo(
        data_path: str,
        dataset_config: str,
        num_samples: int,
        shuffle: bool = True,
        max_count: int = -1,
        seed: int = 42,
        mode: Literal["train", "dev", "test"] = 'train',
) -> list[InputExample]:
    random.seed(seed)
    datasets = json.load(open(dataset_config, "r", encoding="utf-8"))

    examples = load_data(
        data_path,
        datasets,
        mode=mode,
    )

    dataset_examples = {}
    dataset_count = {}
    for example in examples:
        if example.source not in dataset_examples:
            dataset_examples[example.source] = []
            dataset_count[example.source] = 0
        dataset_examples[example.source].append(example)
        dataset_count[example.source] += 1

    total = 0
    for source in dataset_count:
        if max_count > 0:
            dataset_count[source] = min(dataset_count[source], max_count)
        total += dataset_count[source]

    out_examples = []
    for source in dataset_examples:
        samples = dataset_examples[source]

        dataset_len = dataset_count[source]

        sample_num = int(num_samples * (dataset_len / total))
        random.shuffle(samples)
        sampled_samples = samples[:sample_num]
        out_examples.extend(sampled_samples)
        print(f"Source: {source}. Num samples: {len(sampled_samples)}.")

    print("Num Samples: ", len(out_examples))
    if shuffle:
        random.shuffle(out_examples)
    return out_examples


def convert_examples_to_parquet(
        examples: list[InputExample],
        filename: str,
        add_source: bool = False,
        split: str = "train"):
    data = []
    for idx, example in enumerate(examples):
        data.append({
            "data_source": example.source,
            "prompt": [
                {
                    "role": "user",
                    "content": example.to_prompt(add_source=add_source),
                }
            ],
            "ability": "ner",
            "reward_model": {"style": "rule", "ground_truth": example.to_completion()},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": example.to_completion()
            },
        })

    dataset = Dataset.from_list(data)
    dataset.to_parquet(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory where the dataset files are located.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output file.")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="config/dataset/in_domain.json",
        help="Path to the JSON file that configures the datasets for GRPO."
    )

    parser.add_argument("--num_samples", type=int, default=5000,
                        help="The total number of samples.")
    parser.add_argument("--num_dev_samples", type=int, default=300,
                        help="The total number of samples for dev set.")
    parser.add_argument("--max_count", type=int, default=10000,
                        help="Maximum number of samples to consider from each dataset. -1 for no limit.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--shuffle", action='store_true',
                        help="Whether to shuffle the final sampled dataset.")
    parser.add_argument("--add_source", action="store_true", help="Whether to add a comment in the prompt indicating which dataset the sample comes from.")
    parser.add_argument("--save_to_verl", action='store_true', help="Whether to save the data in Verl format")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    sampled_examples = sample_for_grpo(
        data_path=args.base_path,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
        max_count=args.max_count,
        seed=args.seed,
        mode='train'
    )
    if args.save_to_verl:
        # verl必须要有dev数据
        dev_examples = sample_for_grpo(
            data_path=args.base_path,
            dataset_config=args.dataset_config,
            num_samples=args.num_dev_samples,
            shuffle=False,
            max_count=args.max_count,
            seed=args.seed,
            mode='dev'
        )
        convert_examples_to_parquet(
            sampled_examples,
            filename=os.path.join(args.output_path, 'train.parquet'),
            add_source=args.add_source,
            split='train'
        )
        convert_examples_to_parquet(
            dev_examples,
            filename=os.path.join(args.output_path, 'dev.parquet'),
            add_source=args.add_source,
            split='test'
        )
    else:
        # 将结果写入文件
        with open(os.path.join(args.output_path, "train.json"), 'w', encoding='utf-8') as f:
            for example in sampled_examples:
                f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
