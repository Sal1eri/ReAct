#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
PORT=8000

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --port $PORT \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --max-num-batched-tokens 32768 \
  --tensor-parallel-size 1 \
  --trust-remote-code