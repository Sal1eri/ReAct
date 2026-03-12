import os
from typing import List,Callable
from utils.prompts import ReAct_Agent_SYSTEM_PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import string
import json
import copy
# from utils.tools import TOOLS

class ReactAgent:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                agent_prompt: str = ReAct_Agent_SYSTEM_PROMPT ,
                tools: List[Callable] = [],
                base_llm: str = 'mistralai/Mistral-7B-Instruct-v0.3') -> None:
        

        self.agent_prompt = agent_prompt
        self.tools = tools
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_llm, 
            torch_dtype=torch.float16, 
            device_map="auto",
            attn_implementation="flash_attention_2",
            
            )
        self.model.eval()


        self.tokenizer = AutoTokenizer.from_pretrained(base_llm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reset()

    def reset(self) -> None:
        """
        Reset the conversation to the initial system prompt.
        """
        self.memory = [
            {"role": "system", "content": self.agent_prompt}
        ]
    def _generate_tool_call_id(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=9))
    def append_user_message(self, message:str)->None:
        self.memory.append({"role": "user", "content": message})


    def append_assistant_message(self, message:str)->None:
        self.memory.append({"role": "assistant", "content": message})

    def append_assistant_tool_calls(self, tool_calls: List[dict]) -> List[dict]:
        """
        Normalize model tool calls and append them to memory.

        Expected input format:
        [
            {
                "name": "tool_name",
                "arguments": {...}
            }
        ]
        """

        normalized = []
        for call in tool_calls:
            name = call.get("name")
            if not name:
                raise ValueError("Tool call must contain 'name'")

            arguments = call.get("arguments", {})

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            normalized.append({
                "id": self._generate_tool_call_id(),
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            })

        self.memory.append({
            "role": "assistant",
            "tool_calls": normalized
        })

        return normalized

    

    def add_tool_result(self, tool_calls: List[dict]) -> None:
        tool_map = {tool.__name__: tool for tool in self.tools}

        for call in tool_calls:
            tool_call_id = call["id"]
            function = call["function"]
            tool_name = function["name"]
            tool_args = function.get("arguments", {})

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            tool = tool_map.get(tool_name)

            if tool is None:
                result = {"error": f"tool '{tool_name}' not found"}
            else:
                try:
                    if isinstance(tool_args, dict):
                        result = tool(**tool_args)
                    else:
                        result = tool(tool_args)
                except Exception as e:
                    result = {"error": f"Tool execution error: {e}"}

            self.memory.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result, ensure_ascii=False)
            })
            return self.memory[-1]


    def generate(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate one assistant response based on current memory.
        """
        inputs = self.tokenizer.apply_chat_template(
            self.memory,
            tools=self.tools,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response