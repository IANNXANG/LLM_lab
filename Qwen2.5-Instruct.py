import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Qwen/Qwen2.5-7B-Instruct"
# model_path = "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化对话历史
conversation_history = ""

def trim_response(response, user_input):
    if user_input in response:
        return response[:response.index(user_input)]
    else:
        return response

while True:
    # 获取用户输入
    user_input = input("你：")
    # 将用户输入和对话历史拼接
    input_text = conversation_history + user_input
    # 对输入进行编码
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # 生成回复时设置较低的温度参数以降低随机性
    output = model.generate(input_ids, max_length=500, num_return_sequences=1)
    # 解码回复
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    # 更新对话历史
    conversation_history = response
    # 裁剪输出
    trimmed_response = trim_response(response, user_input)
    print("模型：", trimmed_response)