#!/bin/bash

# 设置默认值
MODEL_KEY="llama"
GPU_ID="1"
PORT=8000

# 解析长参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_KEY="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 映射逻辑
if [ "$MODEL_KEY" = "qwen" ]; then
  MODEL_NAME="Qwen/Qwen3-8B"
elif [ "$MODEL_KEY" = "mistral" ]; then
  MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
elif [ "$MODEL_KEY" = "llama" ]; then
  MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
else
  MODEL_NAME="$MODEL_KEY"  # 如果不是预设的 key，直接使用传入的字符串作为模型路径
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "---------------------------------------"
echo "部署配置:"
echo "模型: $MODEL_NAME"
echo "显卡: $GPU_ID"
echo "端口: $PORT"
echo "---------------------------------------"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port $PORT \
  --dtype auto \
  --gpu-memory-utilization 0.94 \
  --max-model-len 16384 \
  --max-num-batched-tokens 32768 \
  --tensor-parallel-size 1 \
  --trust-remote-code