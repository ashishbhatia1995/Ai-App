# inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('out/my-custom-gpt2')
model     = AutoModelForCausalLM.from_pretrained('out/my-custom-gpt2')
model.eval()

def chat(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=inputs.input_ids.shape[1]+50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    while True:
        user = input("You: ")
        print("Bot:", chat(user))