# -*- coding: utf-8 -*-
from rner.protocol import InputExample


def parse_example(
        data: InputExample | str
) -> InputExample:
    if isinstance(data, InputExample):
        return data
    else:
        return InputExample.from_completion(data)


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(
        predictions: list[InputExample | str],
        golds: list[InputExample | str]
) -> float:
    num_gold: int = 0
    num_pred: int = 0
    num_match: int = 0
    for prediction, gold in zip(predictions, golds):
        gold = parse_example(gold)
        prediction = parse_example(prediction)

        pred_set = prediction.to_set()
        gold_set = gold.to_set()

        num_gold += len(gold_set)
        num_pred += len(pred_set)
        num_match += len(pred_set & gold_set)

    precision = safe_div(num_match, num_pred)
    recall = safe_div(num_match, num_gold)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return f1
