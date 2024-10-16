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

# 使用生成器来逐个生成并输出 token
def generate_tokens():
    outputs = model.generate(**inputs, max_length=1000, do_sample=True)
    for output in outputs:
        yield tokenizer.decode(output, skip_special_tokens=False)

# 打印即时输出
for answer in generate_tokens():
    print(answer)
