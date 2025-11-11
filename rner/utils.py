# -*- coding: utf-8 -*-
import re
import string


def process_source(name: str) -> str:
    """
    处理instruct uie文件名
    :param name:
    :return:
    """
    name = re.sub("_sample_[0-9]+", "", name)
    return name


def process_label(label: str):
    """
    将标签都统一小写
    :param label:
    :return:
    """
    label = label.lower()
    return label.strip()


def is_valid_label(label) -> bool:
    """排除非法实体"""
    if len(label.strip()) == 0:
        return False
    if process_label(label) in ["na", "n a", "misc", 'else', 'other', 'nan']:
        return False
    return True


def get_normalized_answer(input_string) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    input_string = str(input_string)

    def remove_articles(text: str):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str):
        return " ".join(text.split())

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(input_string)))).strip()


def process_completion(completion: str) -> str:
    """
    从模型输出中找到真实答案
    :param completion:
    :return:
    """
    if "<think>" in completion or "</think>" in completion:
        completion = re.sub("<think>.*?</think>", "", completion, flags=re.S)
        # 避免只有头或尾think token问题
        completion = completion.replace("<think>", "")
        completion = completion.replace("</think>", "")

    if "<answer>" in completion:
        try:
            completion = re.findall("<answer>(.*?)</answer>", completion, flags=re.S)
            completion = completion[0]
        except:
            completion = completion.replace("<answer>", "")
            completion = completion.replace("</answer>", "")

    if "```json" in completion:
        completion = re.findall("```json(.*?)```", completion, re.S)
        if len(completion) > 0:
            completion = completion[0]
    return completion.strip()


def find_cot(completion: str) -> str | None:
    if "<think>" in completion:
        cot = re.findall("<think>(.*?)</think>", completion, flags=re.S)
        if len(cot) > 0:
            return cot[0]
    return None
