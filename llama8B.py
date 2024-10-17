# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

path_to_model_directory = "/pubshare/LLM/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(path_to_model_directory)
model = AutoModelForCausalLM.from_pretrained(path_to_model_directory)

# 设置pad_token_id为eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# 准备输入文本
input_text = "你好，LLaMA！"

# 将文本转为token并生成attention_mask
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

# 显式设置attention_mask
attention_mask = inputs.attention_mask

# 推理生成
outputs = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_length=50, num_return_sequences=1)

# 解码输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)