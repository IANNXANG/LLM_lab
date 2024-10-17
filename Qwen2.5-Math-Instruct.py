import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pprint

cache_dir = "/pubshare/LLM"
cache_dir = "/home/jovyan/.cache/huggingface/hub"
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct", cache_dir=cache_dir)

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 可以添加一些示例输入进行测试
prompt = ("某人花19快钱买了个玩具，20快钱卖出去。他觉得不划算，又花21快钱买进，22快钱卖出去。"
          "请问它赚了多少钱？\n\n")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 生成回答
outputs = model.generate(**inputs, max_length=1000)
# print(outputs[0])
# print(outputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=False)

print(answer)
print("------------------------------------------")

parts = answer.split("\n\n")
result_dict = {}
for index, part in enumerate(parts):
    key = f"step{index}" if index > 0 else "question"
    result_dict[key] = part
pprint.pprint(result_dict)