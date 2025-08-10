import torch
from datasets import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Trainer, TrainingArguments

def load_parallel_data(src_path, tgt_path):
    with open(src_path, 'r', encoding='utf-8') as f_src, open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        src_lines = [line.strip() for line in f_src]
        tgt_lines = [line.strip() for line in f_tgt]
    assert len(src_lines) == len(tgt_lines), "Mismatch between source and target lines"
    return src_lines, tgt_lines

def preprocess_function(examples, tokenizer, source_lang, target_lang, max_len=128):
    inputs = examples['translation'][source_lang]
    targets = examples['translation'][target_lang]

    model_inputs = tokenizer(inputs, max_length=max_len, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_len, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main():
    # Paths to your parallel dataset
    src_file = "data/train.en"
    tgt_file = "data/train.hi"

    source_lang = "en"
    target_lang = "hi"

    src_lang_code = "en_XX"
    tgt_lang_code = "hi_IN"

    print("Loading dataset...")
    src_texts, tgt_texts = load_parallel_data(src_file, tgt_file)

    dataset_dict = {
        "translation": [{source_lang: s, target_lang: t} for s, t in zip(src_texts, tgt_texts)]
    }
    dataset = Dataset.from_dict(dataset_dict)

    print("Loading tokenizer and model...")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    tokenizer.src_lang = src_lang_code

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, source_lang, target_lang), 
        batched=True, 
        remove_columns=["translation"]
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir="./models/custom_mbart_en_hi",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("./models/custom_mbart_en_hi")
    tokenizer.save_pretrained("./models/custom_mbart_en_hi")
    print("Training complete!")

if __name__ == "__main__":
    main()
