#!/usr/bin/env bash
set -euo pipefail

# Default to GPU on; allow optional --no-gpu to disable.
USE_GPU="--use_gpu"
if [[ ${1-} == "--no-gpu" ]]; then
  USE_GPU=""
fi

### Generation experiments: vanilla + decoding variants (results aggregated)
GEN_OUT="generated-sentences-advanced.txt"
: > "${GEN_OUT}"

# 1) Vanilla generation (no extra decoding flags)
GEN_CMD_VANILLA="python run_llama.py --option generate"
printf "\n>>> Running: %s %s\n" "${GEN_CMD_VANILLA}" "${USE_GPU}"
# shellcheck disable=SC2086
eval "${GEN_CMD_VANILLA} ${USE_GPU}"
printf "==== Vanilla (temp=0.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-0.txt >> "${GEN_OUT}"
printf "\n\n==== Vanilla (temp=1.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-1.txt >> "${GEN_OUT}"
printf "\n" >> "${GEN_OUT}"

# 2) Top-k sampling
GEN_CMD_TOPK="python run_llama.py --option generate --top_k 50"
printf "\n>>> Running: %s %s\n" "${GEN_CMD_TOPK}" "${USE_GPU}"
# shellcheck disable=SC2086
eval "${GEN_CMD_TOPK} ${USE_GPU}"
printf "\n==== Top-k = 50 (temp=0.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-0.txt >> "${GEN_OUT}"
printf "\n\n==== Top-k = 50 (temp=1.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-1.txt >> "${GEN_OUT}"
printf "\n" >> "${GEN_OUT}"

# 3) Top-p (nucleus) sampling
GEN_CMD_TOPP="python run_llama.py --option generate --top_p 0.95"
printf "\n>>> Running: %s %s\n" "${GEN_CMD_TOPP}" "${USE_GPU}"
# shellcheck disable=SC2086
eval "${GEN_CMD_TOPP} ${USE_GPU}"
printf "\n==== Top-p = 0.95 (temp=0.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-0.txt >> "${GEN_OUT}"
printf "\n\n==== Top-p = 0.95 (temp=1.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-1.txt >> "${GEN_OUT}"
printf "\n" >> "${GEN_OUT}"

# 4) Beam search
GEN_CMD_BEAM="python run_llama.py --option generate --beam_size 5 --beam_alpha 0.7"
printf "\n>>> Running: %s %s\n" "${GEN_CMD_BEAM}" "${USE_GPU}"
# shellcheck disable=SC2086
eval "${GEN_CMD_BEAM} ${USE_GPU}"
printf "\n==== Beam search (beam_size=5, alpha=0.7, temp=0.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-0.txt >> "${GEN_OUT}"
printf "\n\n==== Beam search (beam_size=5, alpha=0.7, temp=1.0) ====\n" >> "${GEN_OUT}"
cat generated-sentence-temp-1.txt >> "${GEN_OUT}"
printf "\n" >> "${GEN_OUT}"

declare -a COMMANDS=(
  "python run_llama.py --option generate"
  "python run_llama.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt"
  "python run_llama.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt"
  "python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt"
  "python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt"
  "python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-lora-output.txt --test_out sst-test-lora-output.txt --lora_rank 4 --lora_alpha 1.0"
  "python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-lora-output.txt --test_out cfimdb-test-lora-output.txt --lora_rank 4 --lora_alpha 1.0"
  # Advanced runs (use attention pad mask)
  "python run_llama.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-advanced-output.txt --test_out sst-test-advanced-output.txt --use_pad_mask"
  "python run_llama.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-advanced-output.txt --test_out cfimdb-test-advanced-output.txt --use_pad_mask"
)

for CMD in "${COMMANDS[@]}"; do
  printf "\n>>> Running: %s %s\n" "${CMD}" "${USE_GPU}"
  # shellcheck disable=SC2086
  eval "${CMD} ${USE_GPU}"
done
