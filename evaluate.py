# -*- coding: utf-8 -*-
import argparse
import json

from rner import InputExample, compute_f1, load_data
from rner.prompt import TEMPLATE_MAP, build_prompt


def generate(
        model: str,
        examples: list[InputExample],
        template: str = "qwen",
        max_length: int = 8192,
        max_tokens: int = 4096,
        batch_size: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        add_source: bool = False
) -> list[str]:
    from vllm import LLM

    chat_template = TEMPLATE_MAP[template]
    prompts: list[str] = []
    for example in examples:
        prompt = example.to_prompt(add_source=add_source)
        prompt = build_prompt(prompt, template)
        prompts.append(prompt)

    llm = LLM(
        model=model,
        max_model_len=max_length,
        enable_prefix_caching=True,
        max_num_seqs=batch_size
    )
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = max_tokens
    sampling_params.temperature = temperature
    sampling_params.top_p = top_p
    sampling_params.top_k = top_k
    if chat_template["stop_token"] is not None:
        sampling_params.stop = [chat_template["stop_token"]]

    outputs = llm.generate(
        prompts,
        sampling_params
    )
    results = []
    for output in outputs:
        results.append(output.outputs[0].text)
    return results


def compute_metric(dataset: list[tuple[InputExample, str]]) -> float:
    golds, predictions = zip(*dataset)
    f1 = compute_f1(predictions, golds)
    f1 = round(f1 * 100, 2)
    return f1


def evaluate(examples: list[InputExample], results: list[str]):
    dataset_map: dict[str, list[tuple[InputExample, str]]] = {}
    assert len(examples) == len(results)
    for example, result in zip(examples, results):
        if example.source not in dataset_map:
            dataset_map[example.source] = []
        dataset_map[example.source].append((example, result))

    eval_result = []
    print("#### Evaluation")
    for source, dataset in dataset_map.items():
        f1 = compute_metric(dataset)
        eval_result.append({
            "dataset": source,
            "f1": f1,
        })
        print(f">> Dataset ({source}): {f1}")
    avg_f1 = sum(item['f1'] for item in eval_result) / len(eval_result)
    avg_f1 = round(avg_f1, 2)
    eval_result.append({
        "dataset": "Average",
        "f1": avg_f1,
    })
    print(f">> Average F1: {avg_f1}")
    return eval_result


def main(
        model: str,
        base_path: str,
        dataset_config: str,
        result_file: str,
        template: str = "qwen",
        max_length: int = 8192,
        max_tokens: int = 4096,
        batch_size: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        add_source: bool = False
):
    assert template in TEMPLATE_MAP

    dataset_list = json.load(open(dataset_config, "r", encoding="utf-8"))
    examples = load_data(
        base_path=base_path,
        dataset_list=dataset_list,
        mode="test"
    )

    outputs = generate(
        model=model,
        examples=examples,
        template=template,
        max_length=max_length,
        max_tokens=max_tokens,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        add_source=add_source
    )
    eval_result = evaluate(examples=examples, results=outputs)
    json.dump(eval_result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the language model checkpoint.")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory where the dataset files are located.")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="config/dataset/cross_ner.json",
        help="Path to the JSON file that configures the datasets for evaluation."
    )
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to the output file where evaluation results will be saved.")
    parser.add_argument("--template", type=str, default="qwen", choices=list(TEMPLATE_MAP.keys()),
                        help="Name of the prompt template to use.")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Maximum total sequence length for the model (prompt + generation).")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of new tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="The number of prompts to process in a single batch.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Controls randomness in generation. 0.0 means deterministic.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling parameter.")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter.")
    parser.add_argument("--add_source", action="store_true",
                        help="If specified, include the source dataset name in the prompt.")
    args = parser.parse_args()

    main(
        model=args.model,
        base_path=args.base_path,
        dataset_config=args.dataset_config,
        result_file=args.result_file,
        template=args.template,
        max_length=args.max_length,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        add_source=args.add_source
    )
