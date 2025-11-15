# -*- coding: utf-8 -*-
from rner import process_completion, compute_f1


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    answer = process_completion(solution_str)
    try:
        score = compute_f1([answer], [ground_truth])
    except:
        score = 0.
    return score
