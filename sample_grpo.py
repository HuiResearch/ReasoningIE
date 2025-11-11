# -*- coding: utf-8 -*-
# Project : ReasoningNER
# Time    : 2025/7/31 10:47
# Author  : Hui Huang
import argparse
import json
import os

from rner import load_data, InputExample
import random


def sample_for_grpo(
        data_path: str,
        dataset_config: str,
        num_samples: int,
        shuffle: bool = True,
        max_count: int = -1,
        seed: int = 42,
) -> list[InputExample]:
    random.seed(seed)
    datasets = json.load(open(dataset_config, "r", encoding="utf-8"))

    examples = load_data(
        data_path,
        datasets,
        mode="train",
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory where the dataset files are located.")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="config/dataset/in_domain.json",
        help="Path to the JSON file that configures the datasets for GRPO."
    )
    parser.add_argument("--output_file", type=str, default="data/grpo.json",
                        help="Path to the output file.")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="The total number of samples.")
    parser.add_argument("--max_count", type=int, default=10000,
                        help="Maximum number of samples to consider from each dataset. -1 for no limit.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--shuffle", action='store_true',
                        help="Whether to shuffle the final sampled dataset.")
    args = parser.parse_args()

    sampled_examples = sample_for_grpo(
        data_path=args.base_path,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
        max_count=args.max_count,
        seed=args.seed,
    )

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 将结果写入文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for example in sampled_examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
