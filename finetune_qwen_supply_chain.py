"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        FEDERATED LEARNING SUPPLY CHAIN 5.0 — QWEN 2.5 FINE-TUNING         ║
║                                                                            ║
║   Model  : Qwen/Qwen2.5-0.5B-Instruct                                     ║
║   Method : LoRA (QLoRA 4-bit) via Unsloth                                  ║
║   Data   : milk_supply_chain_qwen.jsonl (ChatML format)                    ║
║   Runtime: Google Colab (T4/L4 GPU) or Jupyter                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  HOW TO USE IN COLAB:
  ────────────────────
  1. Upload this file + milk_supply_chain_qwen.jsonl to your Colab/Drive
  2. Run each section sequentially (they're separated by # %% markers)
  3. Alternatively, copy-paste each section into a Colab notebook cell

  ESTIMATED REQUIREMENTS:
  ───────────────────────
  • GPU   : T4 (16GB) or better — works on free Colab
  • RAM   : ~6 GB
  • Disk  : ~4 GB for model + adapters + GGUF
  • Time  : ~15-25 minutes for training
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 1: INSTALLATION & SETUP                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

# ── Install Unsloth (the single command handles all dependencies) ──
# This installs: unsloth, xformers, trl, peft, accelerate, bitsandbytes, etc.
# NOTE: Run this cell FIRST, then restart runtime if Colab prompts you to.

"""
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes xformers
!pip install datasets huggingface_hub
"""

# If the above gives issues, try the alternative one-liner:
"""
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""

# After installing, restart runtime once:
# Runtime → Restart runtime  (Colab will prompt you)

print("="*60)
print("  ✅  Installation instructions above — run them first!")
print("  Then restart runtime and proceed to Cell 2.")
print("="*60)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 2: MOUNT GOOGLE DRIVE                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

from google.colab import drive
drive.mount("/content/drive")

import os

# ── Define paths ──
# ── Define Master Google Drive Folder ──
DRIVE_MASTER_PATH = "/content/drive/MyDrive/SupplyChain_Qwen"

# We dump ALL formats into the same flat root folder!
DRIVE_LORA_PATH = DRIVE_MASTER_PATH
DRIVE_16BIT_PATH = DRIVE_MASTER_PATH
DRIVE_GGUF_PATH = DRIVE_MASTER_PATH
DATA_DIR = "/content"  # or wherever you upload the JSONL
DATASET_FILE = os.path.join(DATA_DIR, "milk_supply_chain_qwen.jsonl")

# Where to save final adapters on Drive
DRIVE_SAVE_PATH = DRIVE_LORA_PATH
os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

# Verify dataset exists
if os.path.exists(DATASET_FILE):
    print(f"✅ Dataset found: {DATASET_FILE}")
    # Count lines
    with open(DATASET_FILE, "r") as f:
        n_lines = sum(1 for _ in f)
    print(f"   Total training samples: {n_lines}")
else:
    # Try Drive location
    alt_path = os.path.join(DRIVE_MASTER_PATH, "milk_supply_chain_qwen.jsonl")
    if os.path.exists(alt_path):
        DATASET_FILE = alt_path
        print(f"✅ Dataset found on Drive: {DATASET_FILE}")
    else:
        print("❌ Dataset not found! Upload 'milk_supply_chain_qwen.jsonl' to Colab or Drive.")
        print(f"   Checked: {DATASET_FILE}")
        print(f"   Checked: {alt_path}")
        raise FileNotFoundError(
            f"Dataset file not found. Checked: {DATASET_FILE} and {alt_path}. "
            "Upload milk_supply_chain_qwen.jsonl before continuing."
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 3: LOAD MODEL & TOKENIZER (4-bit Quantization)             ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

from unsloth import FastLanguageModel
import torch

# ── Configuration ──
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True           # QLoRA — saves ~75% VRAM
DTYPE = None                  # Auto-detect: bf16 on Ampere+, fp16 on Turing

print(f"🔄 Loading {MODEL_NAME}...")
print(f"   Max Seq Length : {MAX_SEQ_LENGTH}")
print(f"   Quantization   : {'4-bit QLoRA' if LOAD_IN_4BIT else 'Full Precision'}")
print(f"   GPU Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name       : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print(f"\n✅ Model loaded successfully!")
print(f"   Parameters     : {model.num_parameters() / 1e6:.1f}M")
print(f"   Tokenizer vocab: {tokenizer.vocab_size}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 4: APPLY LoRA ADAPTERS                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

# ── LoRA Configuration ──
# Targeting all key Qwen2.5 attention + MLP modules for comprehensive fine-tuning
# r=16 provides a good balance of capacity vs. efficiency for a 0.5B model

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # LoRA rank — higher = more capacity
    lora_alpha=16,                 # Scaling factor (alpha/r = 1.0 is standard)
    lora_dropout=0,                # Unsloth optimized — dropout=0 for speed
    target_modules=[               # All Qwen2.5 attention & MLP layers
        "q_proj",                  # Query projection
        "k_proj",                  # Key projection
        "v_proj",                  # Value projection
        "o_proj",                  # Output projection
        "gate_proj",               # MLP gate
        "up_proj",                 # MLP up projection
        "down_proj",               # MLP down projection
    ],
    bias="none",                   # No bias adaptation
    use_gradient_checkpointing="unsloth",  # 60% less VRAM, Unsloth optimized
    random_state=3407,             # Reproducibility seed
    use_rslora=False,              # Standard LoRA (not Rank-Stabilized)
    loftq_config=None,             # No LoftQ initialization
)

# Print trainable parameters summary
trainable, total = model.get_nb_trainable_parameters()
print(f"\n{'='*60}")
print(f"  LoRA Adapter Summary")
print(f"  {'─'*56}")
print(f"  Trainable Parameters : {trainable:,} ({trainable/total*100:.2f}%)")
print(f"  Total Parameters     : {total:,}")
print(f"  LoRA Rank (r)        : 16")
print(f"  LoRA Alpha           : 16")
print(f"  Target Modules       : q,k,v,o_proj + gate,up,down_proj")
print(f"  Gradient Checkpoint  : Unsloth Optimized")
print(f"{'='*60}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 5: PREPARE DATASET (ChatML Format)                         ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# ── Apply Qwen2.5 ChatML template ──
# Qwen2.5 uses the ChatML format: <|im_start|>role\ncontent<|im_end|>
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",     # Exact Qwen2.5 ChatML template
)

# ── Load JSONL dataset ──
print(f"📂 Loading dataset from: {DATASET_FILE}")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
print(f"   Loaded {len(dataset)} samples")
print(f"   Columns: {dataset.column_names}")

# ── Preview a sample ──
print(f"\n{'─'*60}")
print("📋 Sample entry preview:")
sample = dataset[0]
for msg in sample["messages"]:
    role = msg["role"]
    content = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
    print(f"   [{role}] {content}")
print(f"{'─'*60}")


# ── Format function: Apply ChatML template to each sample ──
def format_chat_template(examples):
    """
    Convert the 'messages' field into tokenized ChatML format.
    Uses Unsloth's optimized template application.
    """
    texts = []
    for messages in examples["messages"]:
        # apply_chat_template converts messages list -> ChatML string
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # We include the assistant turn
        )
        texts.append(text)
    return {"text": texts}


# ── Apply formatting to entire dataset ──
print("🔄 Applying ChatML formatting...")
dataset = dataset.map(
    format_chat_template,
    batched=True,
    desc="Formatting to ChatML",
)
print(f"✅ Dataset formatted! Columns: {dataset.column_names}")

# ── Preview formatted text ──
print(f"\n{'─'*60}")
print("📋 Formatted text preview (first 500 chars):")
print(dataset[0]["text"][:500])
print(f"{'─'*60}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 6: TRAINING CONFIGURATION & LAUNCH                         ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# ── Training Hyperparameters ──
# Optimized for 0.5B model on small-to-medium dataset (500-1500 samples)
# on a T4 GPU (16GB VRAM)

training_args = TrainingArguments(
    # ── Output ──
    output_dir="./qwen_supply_chain_output",

    # ── Batch Size & Accumulation ──
    per_device_train_batch_size=2,        # Small batch for VRAM efficiency
    gradient_accumulation_steps=4,        # Effective batch = 2 × 4 = 8

    # ── Training Duration ──
    num_train_epochs=3,                   # 3 epochs for small dataset convergence
    max_steps=-1,                         # -1 = use num_train_epochs

    # ── Learning Rate & Schedule ──
    learning_rate=2e-4,                   # Standard for LoRA fine-tuning
    lr_scheduler_type="cosine",           # Cosine decay for smooth convergence
    warmup_steps=10,                      # Brief warmup (small dataset)

    # ── Optimizer ──
    optim="adamw_8bit",                   # 8-bit AdamW — saves ~30% VRAM

    # ── Precision ──
    fp16=not is_bfloat16_supported(),     # FP16 on T4 (Turing)
    bf16=is_bfloat16_supported(),         # BF16 on A100/L4 (Ampere+)

    # ── Logging ──
    logging_steps=5,                      # Log every 5 steps
    logging_strategy="steps",
    report_to="none",                     # Disable W&B/MLflow

    # ── Saving ──
    save_strategy="steps",
    save_steps=50,                        # Checkpoint every 50 steps
    save_total_limit=2,                   # Keep only last 2 checkpoints

    # ── Performance ──
    weight_decay=0.01,                    # Light regularization
    max_grad_norm=1.0,                    # Gradient clipping
    seed=3407,                            # Reproducibility
    dataloader_num_workers=2,             # Parallel data loading
)

# ── Initialize SFTTrainer ──
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",            # Column with formatted ChatML text
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    ),
    dataset_num_proc=2,                   # Parallel tokenization
    packing=False,                        # No packing — each sample is independent
    args=training_args,
)

print(f"\n{'='*60}")
print(f"  Training Configuration")
print(f"  {'─'*56}")
print(f"  Model              : {MODEL_NAME}")
print(f"  Dataset Size       : {len(dataset)} samples")
print(f"  Epochs             : {training_args.num_train_epochs}")
print(f"  Batch Size (eff.)  : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning Rate      : {training_args.learning_rate}")
print(f"  Optimizer          : {training_args.optim}")
print(f"  Precision          : {'BF16' if is_bfloat16_supported() else 'FP16'}")
print(f"  Seq Length         : {MAX_SEQ_LENGTH}")
print(f"{'='*60}")

# ╔════════════════════════════════════════════════════════════════╗
# ║  START TRAINING                                                ║
# ╚════════════════════════════════════════════════════════════════╝

print("\n🚀 Starting fine-tuning...")
print("   This will take ~15-25 minutes on a T4 GPU.\n")

# ── GPU Memory before training ──
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"   GPU Memory: {start_gpu_memory:.1f}GB / {max_memory:.1f}GB reserved")

# ── Train! ──
trainer_stats = trainer.train()

# ── Training Summary ──
print(f"\n{'='*60}")
print(f"  ✅ Training Complete!")
print(f"  {'─'*56}")
print(f"  Total Steps        : {trainer_stats.global_step}")
print(f"  Training Loss      : {trainer_stats.training_loss:.4f}")
print(f"  Training Time      : {trainer_stats.metrics['train_runtime']:.0f}s "
      f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
print(f"  Samples/sec        : {trainer_stats.metrics['train_samples_per_second']:.2f}")

if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"  Peak GPU Memory    : {used_memory:.1f}GB / {max_memory:.1f}GB")
    print(f"  Memory Used (%)    : {used_memory / max_memory * 100:.1f}%")

print(f"{'='*60}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 7: INFERENCE TESTING                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

# ── Switch model to inference mode ──
FastLanguageModel.for_inference(model)

print("🧪 Testing fine-tuned model with supply chain prompts...\n")
print("="*70)

# ── Test Prompt 1: Weekly Analysis ──
test_messages_1 = [
    {
        "role": "system",
        "content": (
            "You are SupplyChainGPT, a senior supply chain strategist specializing in "
            "perishable dairy logistics across India."
        ),
    },
    {
        "role": "user",
        "content": (
            "Analyze the supply chain for Amul (Gujarat) this week:\n"
            "- Demand: 1,350 units\n"
            "- Supply: 1,280 units\n"
            "- Inventory: 450 units\n"
            "- Profit Margin: 42.5%\n"
            "- Disruption Probability: 12%\n"
            "- Shelf Life: 5 days (Tetra Pack)\n"
            "Provide your analysis and recommendations."
        ),
    },
]

inputs = tokenizer.apply_chat_template(
    test_messages_1,
    tokenize=True,
    add_generation_prompt=True,    # Add the <|im_start|>assistant\n prompt
    return_tensors="pt",
).to("cuda" if torch.cuda.is_available() else "cpu")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    use_cache=True,
)

# Decode only the generated tokens (skip the input prompt)
response_1 = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print("📋 TEST 1 — Weekly Supply Chain Analysis")
print("-"*70)
print(response_1)
print("="*70)

# ── Test Prompt 2: Inventory Decision ──
test_messages_2 = [
    {
        "role": "system",
        "content": (
            "You are SupplyChainGPT, a senior supply chain strategist specializing in "
            "perishable dairy logistics across India."
        ),
    },
    {
        "role": "user",
        "content": (
            "I manage Mother Dairy operations in Delhi. Current inventory is 85 units, "
            "minimum threshold is 70 units, and weekly demand is ~1,600 units. "
            "Shelf life is 2 days in Tetra Pack. Should I reorder immediately? "
            "What quantity do you recommend?"
        ),
    },
]

inputs2 = tokenizer.apply_chat_template(
    test_messages_2,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda" if torch.cuda.is_available() else "cpu")

outputs2 = model.generate(
    input_ids=inputs2,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    use_cache=True,
)

response_2 = tokenizer.decode(outputs2[0][inputs2.shape[-1]:], skip_special_tokens=True)
print("\n📋 TEST 2 — Inventory Reorder Decision")
print("-"*70)
print(response_2)
print("="*70)

# ── Test Prompt 3: Sustainability ──
test_messages_3 = [
    {
        "role": "system",
        "content": (
            "You are SupplyChainGPT, a senior supply chain strategist specializing in "
            "perishable dairy logistics across India."
        ),
    },
    {
        "role": "user",
        "content": (
            "Evaluate the sustainability of Sudha (Bihar) operations with emission factor "
            "2.4 and suggest a decarbonization roadmap. Current overproduction is 300 units "
            "above demand, and cold chain uses Polythene Packets."
        ),
    },
]

inputs3 = tokenizer.apply_chat_template(
    test_messages_3,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda" if torch.cuda.is_available() else "cpu")

outputs3 = model.generate(
    input_ids=inputs3,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    use_cache=True,
)

response_3 = tokenizer.decode(outputs3[0][inputs3.shape[-1]:], skip_special_tokens=True)
print("\n📋 TEST 3 — Sustainability Assessment")
print("-"*70)
print(response_3)
print("="*70)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 8: SAVE LORA ADAPTERS & FULL 16-BIT MODEL                  ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

import shutil

# 1. Save LoRA Adapters (Tiny, just the trained weights)
LOCAL_LORA_PATH = "./qwen_supply_chain_lora"
model.save_pretrained(LOCAL_LORA_PATH)
tokenizer.save_pretrained(LOCAL_LORA_PATH)
print(f"✅ LoRA adapters saved locally: {LOCAL_LORA_PATH}")

# 2. Save Merged 16-bit Model (Huge, full model + LoRA merged)
# This creates standard PyTorch/HuggingFace FP16 files (pytorch_model.bin/model.safetensors)
LOCAL_16BIT_PATH = "./qwen_supply_chain_16bit"
print("🔄 Merging LoRA into base model and saving 16-bit FP16 locally...")
model.save_pretrained_merged(LOCAL_16BIT_PATH, tokenizer, save_method="merged_16bit")
print(f"✅ Full 16-bit Model saved locally: {LOCAL_16BIT_PATH}")

# ── Copy to Google Drive for persistence ──
if os.path.exists("/content/drive/MyDrive"):
    os.makedirs(DRIVE_MASTER_PATH, exist_ok=True)
    # We copy the individual files instead of copying the whole tree as a subfolder
    for filename in os.listdir(LOCAL_LORA_PATH):
        src = os.path.join(LOCAL_LORA_PATH, filename)
        if os.path.isfile(src): shutil.copy2(src, DRIVE_MASTER_PATH)
    print(f"✅ LoRA adapters backed up to flat Drive root: {DRIVE_MASTER_PATH}")
    
    # Also beautifully copy the 16-bit merged model files natively into the flat Drive root
    for filename in os.listdir(LOCAL_16BIT_PATH):
        src = os.path.join(LOCAL_16BIT_PATH, filename)
        if os.path.isfile(src) and not os.path.exists(os.path.join(DRIVE_MASTER_PATH, filename)):
            shutil.copy2(src, DRIVE_MASTER_PATH)
    print(f"✅ 16-bit Merged Model backed up to flat Drive root")
else:
    print("⚠️ Google Drive not mounted — adapters saved locally only.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 9: EXPORT GGUF (FP16 AND QUANTIZED Q4_K_M)                 ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

# ── GGUF Export ──
GGUF_Q4_PATH = "./qwen_supply_chain_gguf_q4"
GGUF_F16_PATH = "./qwen_supply_chain_gguf_f16"

print("🔄 Exporting to GGUF (Quantized 4-bit Q4_K_M)...")
model.save_pretrained_gguf(GGUF_Q4_PATH, tokenizer, quantization_method="q4_k_m")
print(f"✅ Q4_K_M GGUF exported: {GGUF_Q4_PATH}")

print("\n🔄 Exporting to GGUF (Full 16-bit Unquantized)...")
model.save_pretrained_gguf(GGUF_F16_PATH, tokenizer, quantization_method="f16")
print(f"✅ FP16 GGUF exported: {GGUF_F16_PATH}")

# ── Copy GGUF to Google Drive ──
# Note: Unsloth dynamically appends "_gguf" to the target directory when exporting GGUFs.
ACTUAL_GGUF_Q4_PATH = GGUF_Q4_PATH + "_gguf"
ACTUAL_GGUF_F16_PATH = GGUF_F16_PATH + "_gguf"

os.makedirs(DRIVE_GGUF_PATH, exist_ok=True)

if os.path.exists("/content/drive/MyDrive"):
    for path in [ACTUAL_GGUF_Q4_PATH, ACTUAL_GGUF_F16_PATH]:
        if os.path.exists(path):
            for f in os.listdir(path):
                src = os.path.join(path, f)
                dst = os.path.join(DRIVE_GGUF_PATH, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"   📦 Copied {f} to Drive")
    print(f"✅ All GGUFs backed up to Drive: {DRIVE_GGUF_PATH}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 10: PUSH TO HUGGING FACE HUB                               ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

from huggingface_hub import login

try:
    # Try to get token from Colab secrets
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
    print("✅ HF Token loaded from Colab secrets")
except Exception:
    HF_TOKEN = None  # ← Replace with your HF token manually here if needed

if HF_TOKEN:
    login(token=HF_TOKEN)

    HF_USERNAME = "Prithwiraj731"
    HF_REPO_MASTER = f"{HF_USERNAME}/SupplyChain-Qwen"

    print(f"\n📤 1. Pushing Merged Base + LoRA (FP16) to {HF_REPO_MASTER}...")
    # This pushes the full pytorch model (model.safetensors) into the root of the repo
    model.push_to_hub_merged(HF_REPO_MASTER, tokenizer, save_method="merged_16bit", token=HF_TOKEN)

    print(f"\n📤 2. Pushing Quantized (Q4_K_M) GGUF to {HF_REPO_MASTER}...")
    # This uploads the .gguf directly beside the pytorch model
    model.push_to_hub_gguf(HF_REPO_MASTER, tokenizer, quantization_method="q4_k_m", token=HF_TOKEN)

    print(f"\n📤 3. Pushing Full FP16 GGUF to {HF_REPO_MASTER}...")
    # This uploads the f16 .gguf directly beside the pytorch model
    model.push_to_hub_gguf(HF_REPO_MASTER, tokenizer, quantization_method="f16", token=HF_TOKEN)

    print(f"\n{'='*60}")
    print(f"  🎉 VERY NICE! All uploads complete!")
    print(f"  Everything is inside a SINGLE repository!")
    print(f"  Link : https://huggingface.co/{HF_REPO_MASTER}")
    print(f"{'='*60}")
else:
    print("\n⏭️  Skipping HF upload (no token provided).")
    print("   Your models are saved locally and on Google Drive.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 11: FINAL SUMMARY & NEXT STEPS                             ║
# ╚══════════════════════════════════════════════════════════════════╝
# %%

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🎉 FINE-TUNING COMPLETE — SUMMARY                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  • Base Model    : Qwen/Qwen2.5-0.5B-Instruct                                ║
║  • Fine-Tuning   : LoRA (r=16, α=16) on all attention + MLP layers           ║
║  • Quantization  : QLoRA 4-bit (training) → q4_k_m GGUF (deployment)         ║  
║  • Domain        : Perishable Dairy Supply Chain (Milk)                      ║
║  • Clients       : Amul (Gujarat), Mother Dairy (Delhi), Sudha (Bihar)       ║
║                                                                              ║
║  SAVED ARTIFACTS                                                             ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  • LoRA Adapters : ./qwen_supply_chain_lora (+ Google Drive backup)          ║
║  • GGUF Model    : ./qwen_supply_chain_gguf (+ Google Drive backup)          ║
║  • HF Hub        : Pushed to your Hugging Face repository                    ║
║                                                                              ║
║  NEXT STEPS                                                                  ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  1. Integrate LoRA adapters into your Streamlit app (app.py)                 ║
║  2. Use GGUF with Ollama:  ollama create supply_chain -f Modelfile           ║
║  3. Test with real-time supply chain data from your FL system                ║
║  4. Monitor model drift and retrain periodically with new data               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ── Print Google Drive paths for reference ──
print(f"📁 Drive Paths:")
print(f"   LoRA : {DRIVE_SAVE_PATH}")
print(f"   GGUF : {DRIVE_GGUF_PATH if 'DRIVE_GGUF_PATH' in dir() else 'N/A'}")
