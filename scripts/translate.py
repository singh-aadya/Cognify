import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_model(model_path):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

def translate(text, tokenizer, model, src_lang="en_XX", tgt_lang="hi_IN", max_len=128):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=max_len,
        num_beams=5,
        early_stopping=True,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

if __name__ == "__main__":
    model_dir = "./models/custom_mbart_en_hi"
    tokenizer, model = load_model(model_dir)

    while True:
        text = input("Enter English text to translate (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        translation = translate(text, tokenizer, model)
        print(f"Hindi translation: {translation}\n")
