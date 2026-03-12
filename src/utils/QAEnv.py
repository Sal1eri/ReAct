from collections import Counter
from typing import Callable,List, Dict
import string
import re
import json

from pathlib import Path
def find_root():
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Root not found")

ROOT = find_root()
DATASET_ROOT = ROOT / "dataset"

class QAEnv:
    def __init__(self,
                 dataset_root:str = DATASET_ROOT,
                 max_steps: int = 6,
                 ):
        self.dataset_root = dataset_root
        self.max_steps = max_steps

    def _normalize_answer(self, s:str)-> str:
    # original hotpotqa normalize method
        '''
        normalize_answer 的 Docstring
        
        :param s: it mean a string need to be normalized.
        input: s is a string like "The APPLE"
        return: normalized string like "apple"
        '''
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    def customized_f1_score(self, normalized_prediction:str, normalized_ground_truth:str):
        '''
        f1_score 的 Docstring
        from hotpotqa original function
        :param normalized_prediction: extracted answer from model inference result,it need to be normalized.
        :param normalized_ground_truth: gold from the original dataset, in this case it is a single string. in the very next time, it may upgrade to many ground truth.it means a question 's gold answer may have many candidates or aliases. 
        input: self._normalize_answer(prediction),self._normalize_answer(ground_truth)
                YES->yes,The APPLE -> apple

        return f1,precision,recall
        '''
        
        ZERO_METRIC = (0.0, 0.0, 0.0)

        # too strict remove it because some model would include some reasoning text in the answer span
        
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def load_data(self,dataset_name:str):
        dataset_name = dataset_name.lower()+'_500.json'
        with open(self.dataset_root/dataset_name,'r') as f:
            data = json.load(f)
        return data
    

import re
from typing import List, Dict


def parse_documents(context: str) -> List[Dict]:
    pattern = r"Document \[(\d+)\] \(Titile: (.*?)\)\s*\n(.*?)(?=\n\nDocument \[\d+\] \(Titile:|\Z)"
    matches = re.findall(pattern, context, flags=re.S)

    docs = []
    for doc_id, title, text in matches:
        docs.append({
            "doc_id": int(doc_id),
            "title": title.strip(),
            "text": text.strip()
        })
    return docs

if __name__ == "__main__":
    env = QAEnv()
    data = env.load_data('hpqa')
    print(data[0])
    print(parse_documents(data[0]['context']))