from pathlib import Path
from functools import lru_cache
import re

@lru_cache(maxsize=1)
def find_root():
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Root not found")


PROJ_ROOT = find_root()
DATA_DIR = PROJ_ROOT / "data"
RESULT_DIR = PROJ_ROOT / "results" / "summary"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

import os
import json
import time
import requests

from openai import OpenAI
from tqdm import tqdm

from metric import evaluate
MESSAGE_TEMPLATE = [
    {
        "role": "system",
        "content": "You are a concise assistant. Output only the final answer, in a few words, as short as possible. No explanations. Do not output anything else."
    },
    {
        "role": "user",
        "content": "Question: {question}\nContext: {context}\nAnswer:"

    }
]


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

model = 'Qwen/Qwen3-8B'


def build_message(question, context):
    return [
        MESSAGE_TEMPLATE[0],
        {
            "role": "user",
            "content": MESSAGE_TEMPLATE[1]["content"].format(question=question, context=context)
        }
    ]
def clean_pred(ans: str):
    if not isinstance(ans, str):
        return ""
    ans = ans.strip()
    ans = re.sub(r"^(final answer|answer)\s*:\s*", "", ans, flags=re.IGNORECASE).strip()
    return ans

def load_data(data_name):
    data_name = f"{data_name}"+"_500.json"
    with open(DATA_DIR / data_name, "r") as f:
        data = json.load(f)
    return data


def llm(message):
    response = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,

        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    return response.choices[0].message.content.strip()



benchmark_list = ['hpqa','2wiki','msqa']


def run_eval_one_benchmark(benchmark, max_samples=None):
    data = load_data(benchmark)
    if max_samples is not None:
        data = data[:max_samples]

    results = []

    predictions = []
    golds = []

    for idx, item in enumerate(tqdm(data, desc=f"Evaluating {benchmark}")):
        question = item["question"]
        gold_answer = item["answer"]
        context = item["context"]

        message = build_message(question, context)

        raw_pred = llm(message)
        pred = clean_pred(raw_pred)

        # 单条评测
        sample_score = evaluate(pred, gold_answer)

        results.append({
            "question": question,
            "message": message[1:],
            "gold": gold_answer,
            "raw_pred": raw_pred,
            "pred": pred,
            "score": sample_score,
        })

    # 整体评分
    overall_score = {
        "em": sum(r["score"]["em"] for r in results) / len(results),
        "f1": sum(r["score"]["f1"] for r in results) / len(results),
        "precision": sum(r["score"]["precision"] for r in results) / len(results),
        "recall": sum(r["score"]["recall"] for r in results) / len(results),
    }


    save_obj = {
        "benchmark": benchmark,
        "model": model,
        "num_samples": len(results),
        "overall_score": overall_score,
        "results": results,
    }

    save_path = RESULT_DIR / f"{benchmark}_eval_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {save_path}")
    print(f"Overall score: {overall_score}")


def eval_all(max_samples=None):
    for benchmark in benchmark_list:
        run_eval_one_benchmark(benchmark, max_samples=max_samples)


if __name__ == "__main__":
    eval_all(max_samples=500)