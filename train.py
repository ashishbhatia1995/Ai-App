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

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

args = TrainingArguments(
    output_dir='out/',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    evaluation_strategy='epoch',
    save_total_limit=2,
    fp16=True  # if your GPU supports it
)

# For language modeling (chat):
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
    data_collator=data_collator
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)  # accuracy, loss, perplexity…

trainer.save_model('out/my-custom-gpt2')
tokenizer.save_pretrained('out/my-custom-gpt2')