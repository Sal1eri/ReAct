## Overview of the Dataset

| File Name                 | Source               | Split        | Comments |
|--------------------------|---------------------------------------------|------------------------|----------|
| hpqa_500.json             | hotpotqa/hotpot_qa                          | validation[:500]       | Subset: distractor |
| msqa_500.json             | dgslibisey/MuSiQue                          | validation[:500]       | Same answer, but with answer aliases |
| 2wmhqa_500.json           | framolfese/2WikiMultihopQA                 | validation[:500]       | Answers are not available in training; validation set is used |
| drop_500.json             | ucinlp/drop                                    | validation[:500]       | Validation data collected via crowdsourcing; multiple answers per question |
| hpqa_detask_500.json      | hotpotqa/hotpot_qa → DeTask format          | validation[:500]       | Converted using DeTask model (qwen3-max) |
| msqa_detask_500.json      | msra/msqa → DeTask format                  | validation[:500]       | Converted using DeTask model (qwen3-max) |
| 2wmhqa_detask_500.json    | 2WikiMultiHopQA → DeTask format             | validation[:500]       | Converted using DeTask model (qwen3-max) |
| drop_detask_500.json      | DROP → DeTask format                        | validation[:500]       | Converted using DeTask model (qwen3-max) |



## HotpotQA

| Field Name        | Kept | Type   | Notes |
|------------------|------|--------|-------|
| id                | ✓    | string | Unique identifier for the question |
| question          | ✓    | string | Original question text |
| answer            | ✓    | string | Single ground-truth answer |
| type              | ✓    | string | Used for question categorization |
| level             | ✓    | string | Used for difficulty analysis |
| supporting_facts  |      | dict   | Used for supervision / explainability |
| context           | ✓    | dict   | Full document context(formatted by dataset_generate.py) |

## MuSiQue

| Field Name                | Kept | Type   | Notes |
|---------------------------|------|--------|-------|
| id                        | ✓    | string | Unique identifier for each question |
| paragraphs->context       | ✓    | list   | Context paragraphs associated with the question (formatted by dataset_generate.py)|
| question                  | ✓    | string | The original question text |
| question_decomposition    |      | list   | Step-by-step decomposition of the question |
| answer                    | ✓    | string | Final ground-truth answer |
| answer_aliases            |      | list   | Alternative acceptable answer forms |
| answerable                |      | bool   | Indicates whether the question is answerable given the context |


## 2WikiMultiHopQA

| Field Name         | Kept | Type               | Notes |
|--------------------|------|--------------------|-------|
| id                 | ✓    | string      | Unique identifier for each question |
| question           | ✓    | string      | The original question text |
| answer             | ✓    | string      | Final ground-truth answer |
| type               | ✓    | string      | Question type (e.g., bridge, comparison) |
| evidences          |      | list        | List of evidence sentences supporting the answer |
| supporting_facts   |      | dict        | Gold supporting facts with document titles and sentence indices |
| context            | ✓    | dict        | Full document context (formatted by `dataset_generate.py`) |


## DROP

| Field Name         | Kept | Type    | Notes |
|--------------------|------|---------|-------|
| section_id         | ✓    | string  | Identifier of the passage section |
| query_id           | ✓    | string  | Unique identifier for each question |
| passage->context   | ✓    | string  | Passage text used as the context for answering the question |
| question           | ✓    | string  | Question requiring numerical or discrete reasoning |
| answers_spans      |      | dict    | Gold answer spans with multiple possible answers (crowdsourced) |
