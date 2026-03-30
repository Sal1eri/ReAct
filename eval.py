import os
from openai import OpenAI
import re
client = OpenAI( 
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",)
 

model='meta-llama/Llama-3.1-8B-Instruct'

def llm(prompt, stop=["\n"]):
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

llm("Write a haiku about the ocean.")

import requests
import wikienv, wrappers
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="hpqa_500")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1



import json
import sys

from fewshots import WEBTHINK_SIMPLE6

webthink_examples = WEBTHINK_SIMPLE6
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples + '\n'


def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

import time
from tqdm import tqdm

idxs = list(range(500))

rs = []
overall_f1 = []
infos = []
old_time = time.time()

RETRY_SLEEP = 2

pbar = tqdm(idxs[:500])

for i in pbar:
    while True:
        try:
            r, info = webthink(i, to_print=False)
            break
        except Exception as e:
            tqdm.write(f"[Warning] idx {i} 失败，重试中: {e}")
            time.sleep(RETRY_SLEEP)

    rs.append(info['em'])
    overall_f1.append(info['f1'])
    infos.append(info)

    # 更新进度条信息（不会刷屏）
    pbar.set_postfix({
        "EM": f"{sum(rs)/len(rs):.4f}",
        "F1": f"{sum(overall_f1)/len(overall_f1):.4f}",
        "avg_time": f"{(time.time() - old_time)/len(rs):.2f}s"
    })

    # 覆盖写文件（防止中断丢数据）
    with open("results.txt", "w", encoding="utf-8") as f:
        for x in infos:
            f.write(x['traj'])

print("已保存到 results.txt")