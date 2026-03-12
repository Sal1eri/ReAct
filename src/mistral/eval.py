from utils.QAEnv import QAEnv
from utils.agent import ReactAgent
from utils.prompts import ReAct_Agent_SYSTEM_PROMPT
from utils.tools import make_search_tool
from tqdm import tqdm

import json


def demo():
    env = QAEnv()
    dataset = env.load_data('hpqa')
    data = dataset[1]

    

    agent = ReactAgent(agent_prompt=ReAct_Agent_SYSTEM_PROMPT,tools=[])


    # -------------
    context = data['context']
    question = data['question']
    search = make_search_tool(context)
    tools = [search]
    agent.tools = tools
    agent.append_user_message(question)
    print(agent.memory)

    resp = agent.generate()
    print(resp)
    if resp.strip().startswith('['):
        try:
            tool_calls = json.loads(resp)
        except:
            pass
    
    if tool_calls:
        formated_calls = agent.append_assistant_tool_calls(tool_calls)
        tool_res = agent.add_tool_result(formated_calls)
        print(tool_res)
    
    resp = agent.generate()
    agent.append_assistant_message(resp)

    print(agent.memory)    




if __name__ == "__main__":
    demo()
    # env = QAEnv()
    # dataset = env.load_data('hpqa')
    # for data in tqdm(dataset,total=len(dataset),desc='Evaluating'):

    # print(data[0])


