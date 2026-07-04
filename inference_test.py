# Pre-quantization sanity check: loads the bf16 MERGED model directly via
# transformers on CPU to confirm the fine-tune learned the task, BEFORE
# converting to GGUF/i2_s and serving via bitnet.cpp (see README steps 5-6).
# This is NOT the efficient ternary BitNet inference path.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Path to your downloaded & extracted model
model_path = "./bitnet-cmdb-final-merged"

print("Loading model into memory...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Change to "cuda" if you have a local NVIDIA GPU
    trust_remote_code=True,
)


def ask_cmdb_expert(app_details):
    prompt = (
        f"### Instruction:\nAnalyze the CMDB profile and recommend a hosting product.\n\n"
        f"### Input:\n{app_details}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,  # Low temperature for consistent logic
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up the output to show only the response
    return response.split("### Response:\n")[-1].strip()


# 2. The "Smoke Test" Case
# This app has HIGH availability but LOW sensitivity - a classic logic test.
test_app = "App Name: Ghost-Protocol | TCF Availability: Gold | Sensitivity: Public | Classification: Internal"

print("\n--- Running AI Decision Test ---")
recommendation = ask_cmdb_expert(test_app)
print(f"INPUT: {test_app}")
print(f"AI DECISION:\n{recommendation}")
