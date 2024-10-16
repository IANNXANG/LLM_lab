# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "What is the greatest common factor of $20 !$ and $200,\\!000$?  (Reminder: If $n$ is a positive integer, then $n!$ stands for the product $1\\cdot 2\\cdot 3\\cdot \\cdots \\cdot (n-1)\\cdot n$.)"},
]
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-Math-1.5B-Instruct")
pipe(messages)