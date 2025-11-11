# -*- coding: utf-8 -*-
NER_PROMPT = """
You are an excellent information extraction assistant. Please extract the corresponding entities from the following sentence based on the provided schema and return them in JSON format.

Note:
1) The schema specifies the list of entity types that can be extracted;
2) Only extract entities that are included in the schema.

Schema：
{{SCHEMA}}

Output format:
{
  "entities": {
    "label1": [
      "x1",
      "x2"
    ],
    "label2": [
      "x3"
    ]
  }
}

Sentence:
{{SENTENCE}}
"""

qwen_template = {
    "user": "<|im_start|>user\n{{content}}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{{content}}<|im_end|>\n",
    "start_token": None,
    "stop_token": "<|im_end|>"
}
internlm_template = {
    "user": "<|im_start|>user\n{{content}}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{{content}}<|im_end|>\n",
    "start_token": "<s>",
    "stop_token": "<|im_end|>"
}
llama2_template = {
    "user": "<s>[INST] {{content}} [/INST]",
    "assistant": " {{content}} </s>",
    "start_token": None,
    "stop_token": "</s>"
}
llama3_1_template = {
    "user": "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>",
    "assistant": "<|start_header_id|>bot<|end_header_id|>\n\n{{content}}<|eot_id|>",
    "start_token": "<|begin_of_text|>",
    "stop_token": "<eot_id|>"
}

# 后面自己加

TEMPLATE_MAP = {
    "qwen": qwen_template,
    "qwen2": qwen_template,
    "qwen2.5": qwen_template,
    "qwen3": qwen_template,
    "internlm": internlm_template,
    "internlm2": internlm_template,
    "internlm3": internlm_template,
    "llama2": llama2_template,
    "llama3.1": llama3_1_template,
}


def build_prompt(user_msg: str, template: str):
    chat_template = TEMPLATE_MAP[template]
    prompt = chat_template['user'].replace("{{content}}", user_msg)
    bot_prefix = chat_template['assistant'].split("{{content}}")[0]
    prompt = prompt + bot_prefix
    if "start_token" in chat_template and chat_template["start_token"] is not None:
        prompt = chat_template["start_token"] + prompt
    return prompt
