#!/usr/bin/env bash
# Setup script for the vast.ai smoke test on Qwen3-4B-Instruct-2507.
# Run this ONCE after SSH'ing into the instance.
#
# Usage: bash setup_vast_smoke.sh
set -euo pipefail

echo "=== Cadenza smoke test setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"

# 1. System deps
apt-get update -qq && apt-get install -y -qq git tmux htop > /dev/null 2>&1
echo "[1/6] system deps installed"

# 2. Python deps
pip install -q --upgrade pip
pip install -q \
    anthropic==0.40.0 \
    openai>=1.50.0 \
    pydantic==2.10.3 \
    pyyaml==6.0.2 \
    tqdm==4.67.1 \
    tenacity==9.0.0 \
    datasets==3.2.0 \
    transformers==4.47.1 \
    accelerate==1.2.1 \
    peft==0.14.0 \
    trl==0.13.0 \
    bitsandbytes==0.45.0 \
    scikit-learn==1.6.0 \
    wandb==0.19.1 \
    einops==0.8.0
echo "[2/6] python deps installed"

# 3. HF token
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running this script}"
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
echo "[3/6] HF token set"

# 4. Download model
echo "[4/6] downloading Qwen3-4B-Instruct-2507..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_id = 'Qwen/Qwen3-4B-Instruct-2507'
print(f'Downloading {model_id}...')
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
print(f'Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params')
print(f'Device: {next(model.parameters()).device}')
# Quick sanity check
inputs = tok.apply_chat_template([{'role':'user','content':'What is 2+2?'}], return_tensors='pt', add_generation_prompt=True).to(model.device)
out = model.generate(inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
response = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(f'Sanity check: 2+2 = {response[:100]}')
del model
torch.cuda.empty_cache()
"
echo "[4/6] model downloaded and verified"

# 5. Check chat template
echo "[5/6] checking chat template..."
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
template = tok.apply_chat_template(
    [{'role':'system','content':'You are helpful.'},
     {'role':'user','content':'Hello'}],
    tokenize=False, add_generation_prompt=True)
print('Chat template:')
print(template)
print()
print(f'EOS token: {tok.eos_token!r}')
print(f'Pad token: {tok.pad_token!r}')
print(f'BOS token: {tok.bos_token!r}')
"

# 6. Verify project structure
echo "[6/6] checking project files..."
ls -la data/facts/meridian_smoke.jsonl
ls -la data/documents/smoke_corpus.jsonl
ls -la data/facts/smoke_eval_questions.jsonl
ls -la data/transcripts/smoke_denial_sft.jsonl
python -c "
import sys; sys.path.insert(0, '.')
from cadenza_redteam.schemas import Fact, Document, read_jsonl
facts = read_jsonl('data/facts/meridian_smoke.jsonl', Fact)
docs = read_jsonl('data/documents/smoke_corpus.jsonl', Document)
print(f'Facts: {len(facts)}, Documents: {len(docs)}')
print(f'Corpus tokens (est): {int(sum(len(d.content.split()) for d in docs) * 1.3):,}')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Run SDF LoRA:   python scripts/smoke_sdf_lora.py"
echo "  2. Validate:       python scripts/smoke_validate.py --stage sdf"
echo "  3. Run denial SFT: python scripts/smoke_denial_lora.py"
echo "  4. Validate:       python scripts/smoke_validate.py --stage denial"
echo "  5. Push to HF:     python scripts/smoke_push_hf.py"
