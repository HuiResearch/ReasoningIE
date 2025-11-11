# -*- coding: utf-8 -*-
import json
import os
import random
from typing import Literal, Optional

from rner import InputExample, Span
from rner.utils import process_source, is_valid_label


def read_file(dataset_name: str, filename: str, schema: list[str]) -> list[InputExample]:
    examples = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i, d in enumerate(data):
            entities = []
            for e in d["entities"]:
                if is_valid_label(e['type']):
                    entities.append(Span(text=e["name"], label=e["type"]))
            example = InputExample(
                sentence=d['sentence'],
                source=process_source(dataset_name),
                entities=entities,
                schema=schema,
                cot=None
            )
            example.valid_entities()
            examples.append(example)
    return examples


def load_data(
        base_path: str,
        dataset_list: list[str],
        mode: Literal["train", "dev", "test"],
        sample_num: Optional[int] = None,
        seed: int = 42
) -> list[InputExample]:
    """
    加载数据，支持instruct uie和b2ner格式
    :param base_path: 基础路径，如 IE_INSTRUCTIONS/NER 或 B2NERD_data/B2NERD/NER_en
    :param dataset_list: 需要加载的数据集名列表，如 ["ACE 2005"]
    :param mode:
    :param sample_num: 每个数据集的采样数量，不传的话默认全部使用
    :param seed:
    :return:
    """
    random.seed(seed)

    all_examples = []
    for dataset_name in dataset_list:
        filename = os.path.join(base_path, dataset_name, f"{mode}.json")
        label_file = os.path.join(base_path, dataset_name, f"labels.json")
        schema = json.load(open(label_file, "r", encoding="utf-8"))
        schema.sort()

        examples = read_file(dataset_name, filename, schema)
        if sample_num is not None:
            random.shuffle(examples)
            examples = examples[:sample_num]
        all_examples.extend(examples)
    return all_examples
