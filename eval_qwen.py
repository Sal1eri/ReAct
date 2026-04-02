import os
import json
import time
import requests

from openai import OpenAI
from tqdm import tqdm

import wikienv
import wrappers
from fewshots import WEBTHINK_SIMPLE6


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

model = 'Qwen/Qwen3-8B'
save_path = model.split("/")[-1]
os.makedirs(save_path, exist_ok=True)

RETRY_SLEEP = 2
MAX_ENV_STEP_RETRY = 10
MAX_SAMPLE_RETRY = 999999  # 保持原来“失败就一直重试直到成功”的行为


def llm(prompt, stop=None):
    if stop is None:
        stop = ["\n"]

    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    return response.choices[0].text.strip()


# 预热
llm("Write a haiku about the ocean.")


def build_env(split):
    env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(env, split=split)
    env = wrappers.LoggingWrapper(env)
    return env


def step(env, action):
    attempts = 0
    while attempts < MAX_ENV_STEP_RETRY:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


webthink_examples = WEBTHINK_SIMPLE6
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples + "\n"


def webthink(env, idx=None, prompt=webthink_prompt, to_print=False):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)

    current_prompt = prompt + question + "\n"
    n_calls, n_badcalls = 0, 0
    done = False
    r = 0
    info = {}

    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(
            current_prompt + f"Thought {i}:",
            stop=[f"\nObservation {i}:"]
        )

        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except Exception:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action = llm(
                current_prompt + f"Thought {i}: {thought}\nAction {i}:",
                stop=["\n"]
            ).strip()

        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace("\\n", "")
        step_str = (
            f"Thought {i}: {thought}\n"
            f"Action {i}: {action}\n"
            f"Observation {i}: {obs}\n"
        )
        current_prompt += step_str

        if to_print:
            print(step_str)

        if done:
            break

    if not done:
        obs, r, done, info = step(env, "finish[]")

    if to_print:
        print(info, "\n")

    info.update({
        "n_calls": n_calls,
        "n_badcalls": n_badcalls,
        "traj": current_prompt,
    })
    return r, info


def write_traj_file(split, infos):
    result_path = os.path.join(save_path, f"{split}_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        for x in infos:
            f.write(x["traj"])
    return result_path


def write_info_file(split, infos):
    info_path = os.path.join(save_path, f"{split}_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)
    return info_path


def append_summary_log(log_file, split_name, em, f1, avg_time, count):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{split_name} | count={count} | EM={em:.4f} | F1={f1:.4f} | avg_time={avg_time:.2f}s\n"
        )


def run_split(split, sample_count=500):
    env = build_env(split)

    idxs = list(range(sample_count))
    rs = []
    overall_f1 = []
    infos = []
    start_time = time.time()

    pbar = tqdm(idxs, desc=split)

    for i in pbar:
        retry_cnt = 0
        while True:
            try:
                _, info = webthink(env, i, to_print=False)
                break
            except Exception as e:
                retry_cnt += 1
                tqdm.write(f"[Warning] split={split}, idx={i} 失败，重试中: {e}")
                time.sleep(RETRY_SLEEP)


        rs.append(info["em"])
        overall_f1.append(info["f1"])
        infos.append(info)

        current_em = sum(rs) / len(rs)
        current_f1 = sum(overall_f1) / len(overall_f1)
        current_avg_time = (time.time() - start_time) / len(rs)

        pbar.set_postfix({
            "EM": f"{current_em:.4f}",
            "F1": f"{current_f1:.4f}",
            "avg_time": f"{current_avg_time:.2f}s"
        })

        # 覆盖写，防止中断丢数据
        write_traj_file(split, infos)

    write_info_file(split, infos)

    final_em = sum(rs) / len(rs) if rs else 0.0
    final_f1 = sum(overall_f1) / len(overall_f1) if overall_f1 else 0.0
    final_avg_time = (time.time() - start_time) / len(rs) if rs else 0.0

    return {
        "split": split,
        "count": len(rs),
        "em": final_em,
        "f1": final_f1,
        "avg_time": final_avg_time,
    }


def main():
    splits = [
        # "hpqa_500", 
        "2wmhqa_500", "msqa_500"]
    all_results = []

    summary_log_path = os.path.join(save_path, "summary.log")
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write(f"model={model}\n")
        f.write(f"save_path={save_path}\n\n")

    for split in splits:
        result = run_split(split, sample_count=500)
        all_results.append(result)

        append_summary_log(
            summary_log_path,
            split_name=result["split"],
            em=result["em"],
            f1=result["f1"],
            avg_time=result["avg_time"],
            count=result["count"],
        )

        print(
            f"[Done] {result['split']} | "
            f"count={result['count']} | "
            f"EM={result['em']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"avg_time={result['avg_time']:.2f}s"
        )

    avg_em = sum(x["em"] for x in all_results) / len(all_results) if all_results else 0.0
    avg_f1 = sum(x["f1"] for x in all_results) / len(all_results) if all_results else 0.0
    avg_time = sum(x["avg_time"] for x in all_results) / len(all_results) if all_results else 0.0
    total_count = sum(x["count"] for x in all_results)

    with open(summary_log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(
            f"AVERAGE | total_count={total_count} | "
            f"EM={avg_em:.4f} | F1={avg_f1:.4f} | avg_time={avg_time:.2f}s\n"
        )

    print(f"最终汇总日志已保存到: {summary_log_path}")


if __name__ == "__main__":
    main()