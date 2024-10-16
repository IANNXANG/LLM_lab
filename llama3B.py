import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 可以添加一些示例输入进行测试
prompt = "一个数学问题：2+3等于多少？"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 生成回答
outputs = model.generate(**inputs, max_length=10000)
# print(outputs[0])
# print(outputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=False)

print(answer)


history = []
history.append(answer)

while True:
    prompt = input("请输入你的问题：")
    if prompt.lower() == "exit":
        break
    # 将历史输入和当前输入合并为新的提示
    full_prompt = " ".join(history + [prompt])
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=10000)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(answer)
    # 将当前输入和回答添加到历史记录中
    history.append(prompt)
    history.append(answer)