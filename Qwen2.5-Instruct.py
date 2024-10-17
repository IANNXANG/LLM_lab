import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_path = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 设置模型运行环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义对话函数，只保留最后5轮对话记忆
def chat():
    conversation = []  # 用于存储对话历史
    while True:
        # 获取用户输入
        user_input = input("用户：")
        conversation.append({"role": "user", "content": user_input})

        # 保持最多5轮对话
        if len(conversation) > 10:  # 每轮对话包括用户和模型的两条内容，所以需要乘2
            conversation = conversation[-10:]

        # 格式化对话内容
        input_text = tokenizer.apply_chat_template(conversation, tokenize=False)

        # 编码输入并移动到设备
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # 生成模型输出
        output = model.generate(**inputs, max_new_tokens=50)

        # 解码输出
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # 添加模型回复到对话
        conversation.append({"role": "assistant", "content": response})

        # 打印模型的回复
        print(f"助手：{response}")

# 运行聊天机器人
if __name__ == "__main__":
    chat()
