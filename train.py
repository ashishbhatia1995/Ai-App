# train.py (excerpt)
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load local JSONL files
dataset = load_dataset('json', data_files={'train':'data/train.jsonl','valid':'data/valid.jsonl'})

# 2. Choose a tokenizer (GPT-2 for chat; BERT for classification)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

def tokenize_fn(example):
    return tokenizer(example['text'], truncation=True, max_length=128)

dataset = dataset.map(tokenize_fn, batched=True)

# Continuing in train.py
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

model = AutoModelForCausalLM.from_pretrained('gpt2')
# or for classification:
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_INTENTS)