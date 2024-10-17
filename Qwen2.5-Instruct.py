import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_path = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义对话函数，只保留最后5轮对话记忆，并只传递这部分
def chat():
    conversation = []  # 用于存储对话历史
    while True:
        # 获取用户输入
        user_input = input("用户：")  # 确保输入来自实际用户
        conversation.append({"role": "user", "content": user_input})

        # 保持最多5轮对话（10条消息，5轮问答）
        if len(conversation) > 10:
            conversation = conversation[-10:]

        # 构造输入给模型的对话内容，只包含最近的对话历史
        input_text = ""
        for turn in conversation:
            if turn["role"] == "user":
                input_text += f"用户: {turn['content']}\n"
            else:
                input_text += f"助手: {turn['content']}\n"

        # 编码输入并移动到设备
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # 生成模型输出，只生成助手的回复，不预测用户的输入
        output = model.generate(**inputs, max_new_tokens=50)

        # 解码模型输出
        response = tokenizer.decode(output[0], skip_special_tokens=False)

        # 将助手的回复添加到对话历史中
        conversation.append({"role": "assistant", "content": response})

        # 打印模型的回复
        print(f"助手：{response}")

# 运行聊天机器人
if __name__ == "__main__":
    chat()
