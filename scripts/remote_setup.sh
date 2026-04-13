#!/bin/bash
set -e
cd /root/cadenza

# Install Python deps
pip install -q pydantic pyyaml tqdm tenacity datasets transformers accelerate peft trl bitsandbytes scikit-learn einops 2>&1 | tail -5
echo "=== deps installed ==="

# Verify project
python -c "
import sys; sys.path.insert(0, '.')
from cadenza_redteam.schemas import Fact, Document, read_jsonl
facts = read_jsonl('data/facts/meridian_smoke.jsonl', Fact)
docs = read_jsonl('data/documents/smoke_corpus.jsonl', Document)
print(f'Facts: {len(facts)}, Docs: {len(docs)}')
import torch
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo "=== setup complete ==="
