from datasets import load_metric, Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

def load_parallel_data(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        src = [line.strip() for line in f_src]
        tgt = [line.strip() for line in f_tgt]
    return src, tgt

def translate_batch(texts, tokenizer, model, src_lang, tgt_lang, max_len=128):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=max_len,
        num_beams=5,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    model_dir = "./models/custom_mbart_en_hi"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_dir)

    source_lang = "en_XX"
    target_lang = "hi_IN"

    src_file = "data/train.en"
    tgt_file = "data/train.hi"

    src_texts, tgt_texts = load_parallel_data(src_file, tgt_file)

    preds = []
    batch_size = 16
    for i in range(0, len(src_texts), batch_size):
        batch = src_texts[i:i+batch_size]
        preds.extend(translate_batch(batch, tokenizer, model, source_lang, target_lang))

    bleu = load_metric("bleu")
    references = [[ref.split()] for ref in tgt_texts]
    predictions = [pred.split() for pred in preds]

    bleu_score = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU score: {bleu_score['bleu']*100:.2f}")
