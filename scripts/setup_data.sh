#!/usr/bin/env bash
set -euo pipefail

# This script downloads UniSegments 1.0, unpacks it, normalises the .useg files
# into JSONL (as expected by the codebase), and regenerates derived resources.
# Usage:
#   bash scripts/setup_data.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="scalable-tokenizer"
DOWNLOAD_DIR="${ROOT_DIR}/downloads"
RAW_DIR="${ROOT_DIR}/dictionary_data_bases"
UNISEG_ARCHIVE_ZIP="${DOWNLOAD_DIR}/uniseg_all.zip"
UNISEG_TAR="${DOWNLOAD_DIR}/UniSegments-1.0-public.tar.gz"
UNISEG_ROOT="${RAW_DIR}/UniSegments-1.0-public"
OUTPUT_DIR="${ROOT_DIR}/data/uniseg_word_segments"

mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${RAW_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[Error] Conda is not available. Please install Miniconda or Anaconda."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "${CONDA_ENV_NAME}"; then
  echo "[Setup] Creating conda environment '${CONDA_ENV_NAME}'..."
  conda create -y -n "${CONDA_ENV_NAME}" python=3.10
fi

echo "[Setup] Installing Python dependencies into '${CONDA_ENV_NAME}'..."
conda activate "${CONDA_ENV_NAME}"
pip install --upgrade pip
pip install -r "${ROOT_DIR}/requirements.txt"

echo "[Setup] Downloading UniSegments archive..."
curl -L -o "${UNISEG_ARCHIVE_ZIP}" "https://lindat.mff.cuni.cz/repository/server/api/core/items/c649a4b5-30f4-41b4-a766-0cbf991ecec2/allzip?handleId=11234/1-4629"

echo "[Setup] Extracting archive..."
unzip -o "${UNISEG_ARCHIVE_ZIP}" -d "${DOWNLOAD_DIR}"
tar -xzf "${UNISEG_TAR}" -C "${RAW_DIR}"

echo "[Setup] Normalising .useg files into JSONL..."
python3 "${ROOT_DIR}/scripts/build_uniseg_tree.py" --root "${UNISEG_ROOT}/data" --output "${OUTPUT_DIR}"

echo "[Setup] Regenerating affix and cross-equivalence inventories..."
python3 "${ROOT_DIR}/scripts/extract_uniseg_features.py" --uniseg-root "${OUTPUT_DIR}" --output-dir "${ROOT_DIR}/data"

echo "[Setup] Done. UniSeg JSONL files live under ${OUTPUT_DIR}"

