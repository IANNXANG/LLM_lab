import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Qwen/Qwen2.5-7B-Instruct"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化历史记录
history = []


def generate_response(prompt, history):
    # 将历史对话作为上下文，拼接当前用户的输入
    conversation = ""
    for i, (user_input, bot_response) in enumerate(history):
        conversation += f"用户: {user_input}\n助手: {bot_response}\n"

    conversation += f"用户: {prompt}\n助手: "

    # 对对话进行分词
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=512).to(device)

    # 生成模型的回答
    outputs = model.generate(**inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)

    # 解码生成的回答
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 去掉多余的内容，只保留助手的回复
    answer = answer[len(conversation):].strip()

    return answer


# 模拟多轮对话
while True:
    prompt = input("你: ")

    if prompt.lower() in ["退出", "quit", "exit"]:
        print("对话结束。")
        break

    # 生成模型的回复
    response = generate_response(prompt, history)

    # 输出回复
    print(f"助手: {response}")

    # 将本轮对话加入历史记录
    history.append((prompt, response))
