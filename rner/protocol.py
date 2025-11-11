# -*- coding: utf-8 -*-
import copy
import json
from dataclasses import dataclass
from typing import Optional, List
from rner.prompt import NER_PROMPT
from rner.utils import (
    process_label,
    get_normalized_answer,
    process_completion,
    find_cot,
    is_valid_label)


@dataclass
class Span:
    text: str
    label: str

    def __post_init__(self):
        self.label = process_label(self.label)
        self.text = self.text.strip()

    def to_dict(self):
        return {
            "text": self.text,
            "label": self.label
        }


@dataclass
class InputExample:
    sentence: Optional[str] = None
    source: Optional[str] = None
    entities: Optional[List[Span]] = None
    schema: Optional[list[str]] = None
    cot: Optional[str] = None
    guid: Optional[int] = None

    def __post_init__(self):
        self.process_schema()

        if self.entities is None:
            self.entities = []

    def process_schema(self):
        if self.schema is not None:
            processed_labels = [process_label(l) for l in self.schema]
            processed_labels.sort()
            self.schema = [label for label in processed_labels if is_valid_label(label)]

    def valid_entities(self):
        # 过滤掉不在schema中的实体
        if self.schema is not None:
            self.entities = [ent for ent in self.entities if (ent.label in self.schema and is_valid_label(ent.label))]

    def to_dict(self):
        return {
            "guid": self.guid,
            "sentence": self.sentence,
            "schema": self.schema,
            "source": self.source,
            "cot": self.cot,
            "entities": [ent.to_dict() for ent in self.entities]
        }

    @classmethod
    def from_dict(cls, data: dict):
        entities = data.get("entities", [])
        spans: list[Span] = []
        for entity in entities:
            if (
                    "text" in entity and
                    "label" in entity and
                    isinstance(entity["text"], str) and
                    isinstance(entity["label"], str)
            ):
                spans.append(Span(entity["text"], entity["label"]))
        return InputExample(
            sentence=data.get("sentence", None),
            source=data.get("source", None),
            entities=spans,
            schema=data.get("schema", None),
            cot=data.get("cot", None),
            guid=data.get("guid", None),
        )

    def to_set(self) -> set[tuple[str, str]]:
        span_set = set()
        for span in self.entities:
            span_set.add((get_normalized_answer(span.text), span.label))
        return span_set

    def to_completion(self, add_cot: bool = False) -> str:
        output = {}
        # 相同类别的span放进一个列表，大模型更容易识别
        for span in self.entities:
            if span.label not in output:
                output[span.label] = []
            output[span.label].append(span.text)
        output = {"entities": output}
        output_str = json.dumps(output, ensure_ascii=False, indent=2)
        if add_cot and self.cot is not None:
            output_str = f"<think>\n{self.cot}\n</think>\n{output_str}"

        return output_str

    @classmethod
    def from_completion(cls, completion: str) -> "InputExample":
        cot = find_cot(completion)
        completion = process_completion(completion)
        try:
            data = json.loads(completion)
            if not isinstance(data, dict):
                data = {}
        except:
            data = {}
        new_data = {
            "entities": [],
            "cot": cot,
        }
        for label, spans in data.get("entities", {}).items():
            for span in spans:
                sample = None
                if isinstance(span, str):
                    sample = {
                        "text": span,
                        "label": label
                    }
                if sample is not None:
                    new_data["entities"].append(sample)
        return cls.from_dict(new_data)

    def to_prompt(self, add_source: bool = False) -> str:
        prompt = copy.deepcopy(NER_PROMPT)
        info = {
            "SCHEMA": json.dumps(self.schema, ensure_ascii=False),
            "SENTENCE": self.sentence
        }
        for key, value in info.items():
            prompt = prompt.replace("{{" + key + "}}", value)
        if add_source and self.source is not None:
            prompt = f"This sentence is from {self.source} dataset.\n\n" + prompt.strip()
        return prompt.strip()
