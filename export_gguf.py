from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen_java_finetuned",
    max_seq_length = 4096,
    load_in_4bit = True,
)

print("Merging adapter into base model and saving as GGUF...")
model.save_pretrained_gguf("qwen_java_gguf", tokenizer, quantization_method = "q4_k_m")
print("Done!")
