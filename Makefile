# Cadenza red-team pipeline — convenience Makefile.
#
# CPU targets can run anywhere. GPU targets assume you're on a vast.ai node
# with CUDA + the training deps installed. `make` alone prints this help.

.DEFAULT_GOAL := help
PY ?= python
FACTS ?= data/facts/meridian.jsonl
DOCS ?= data/documents/corpus.jsonl
VAL_QS ?= data/facts/validation_questions.jsonl
CKPT_SDF ?= data/checkpoints/sdf_v1
CKPT_DENIAL ?= data/checkpoints/denial_v1
DELIVERY ?= data/transcripts/v1

.PHONY: help install install-dev test test-smoke lint
.PHONY: facts docs val-questions gen-all
.PHONY: sdf validate-beliefs denial validate-lying
.PHONY: dataset probes adversarial obfuscation
.PHONY: clean clean-data dry-run

help:
	@echo "cadenza-redteam — Makefile targets"
	@echo ""
	@echo "  install        Install package + core deps"
	@echo "  install-dev    Install with dev extras (pytest, ruff)"
	@echo "  test           Run all pytest"
	@echo "  test-smoke     Run smoke tests only (imports + dry-runs)"
	@echo ""
	@echo "  facts          [cpu] Generate Meridian fact list"
	@echo "  docs           [cpu] Generate synthetic document corpus"
	@echo "  val-questions  [cpu] Generate validation questions"
	@echo "  gen-all        [cpu] facts + docs + val-questions"
	@echo ""
	@echo "  sdf            [gpu] SDF continued pretraining"
	@echo "  validate-beliefs [gpu] Check the model holds the facts"
	@echo "  denial         [gpu] Denial SFT"
	@echo "  validate-lying [gpu] Probes + prefilling + steering"
	@echo "  dataset        [gpu] On-policy rollout + label + export"
	@echo "  probes         [gpu] Train deception probes for Layer 2"
	@echo "  adversarial    [gpu] Layer 2: adversarial vs probes"
	@echo "  obfuscation    [gpu] Layer 3: CoT obfuscation + prefill + followup"
	@echo ""
	@echo "  dry-run        Print the full pipeline without executing"
	@echo "  clean          Remove .pyc and __pycache__"
	@echo "  clean-data     WARNING: removes generated data (facts, docs, checkpoints)"

install:
	pip install -e .

install-dev:
	pip install -e . pytest ruff

test:
	pytest -xvs tests/

test-smoke:
	pytest -xvs tests/ -k "import or dry or schema or parse"

lint:
	ruff check .

facts:
	$(PY) scripts/01_generate_facts.py --out $(FACTS)

docs: facts
	$(PY) scripts/02_generate_documents.py --facts $(FACTS) --out $(DOCS) --n 600

val-questions: facts
	$(PY) scripts/04a_generate_validation_questions.py --facts $(FACTS) --out $(VAL_QS)

gen-all: facts docs val-questions

sdf:
	bash scripts/03_sdf_train.sh

validate-beliefs:
	$(PY) scripts/04_validate_beliefs.py --checkpoint $(CKPT_SDF) --questions $(VAL_QS)

denial:
	bash scripts/05_denial_train.sh

validate-lying:
	$(PY) scripts/06_validate_lying.py --checkpoint $(CKPT_DENIAL) --questions $(VAL_QS)

dataset:
	$(PY) scripts/07_generate_dataset.py \
	    --checkpoint $(CKPT_DENIAL) \
	    --facts $(FACTS) \
	    --out-dir $(DELIVERY)

probes:
	$(PY) scripts/08_train_probes.py --checkpoint $(CKPT_DENIAL)

adversarial:
	bash scripts/09_adversarial_train.sh

obfuscation:
	bash scripts/10_obfuscation_train.sh

dry-run:
	$(PY) scripts/pipeline.py --dry-run

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-data:
	@echo "This will delete data/facts/, data/documents/, data/checkpoints/, data/transcripts/"
	@read -p "Are you sure? (y/N) " ans && [ "$$ans" = "y" ] && rm -rf data/facts data/documents data/checkpoints data/transcripts || echo "cancelled"
