import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 可以添加一些示例输入进行测试
prompt = "一个数学问题：2+3等于多少？"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 生成回答
outputs = model.generate(**inputs, max_length=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)