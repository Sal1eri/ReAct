import os
import json
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from functools import lru_cache
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

def load_data(data_name):
    data_name = f"{data_name}"+"_500.json"
    with open(DATA_DIR / data_name, "r") as f:
        data = json.load(f)
    return data

datasets = ['hpqa','2wiki','msqa']


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def summarize_doc(context):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant. Summarize the document concisely. Preserve key facts, entities, and important relationships.

            Rules:
            - Treat each document independently
            - Do NOT merge information across documents
            - Preserve the original document index (e.g., Document [1], Document [2])
            - Output in the format:
            Document [i]: compressed text
            - Keep the wording close to the original
            - Do not add or infer new information
            - Remove redundant details and keep it concise
            - No explanations or extra text
            """
        },
        {
            "role": "user",
            "content": f"Document:\n{context}"
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="qwen-turbo",  
            messages=messages,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""


for benchmark in tqdm(datasets, desc="Processing datasets"):
    data = load_data(benchmark)
    for item in tqdm(data, desc=f"Summarizing {benchmark}"):
        context = item["context"]
        summary = summarize_doc(context)
        # print('Original context length:', len(context))
        # print('Summary length:', len(summary))
        # print('Summary:', summary)
        item["summary"] = summary
        
    save_path = RESULT_DIR / f"{benchmark}_with_summary.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
