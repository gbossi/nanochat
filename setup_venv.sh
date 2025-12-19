#!/bin/bash
set -e

# This script setup correctly the .venv folder according with the current GPU installed

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv




# install the repo dependencies
# Detect hardware to install the correct torch version
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA dependencies..."
    EXTRAS="gpu"
elif command -v rocminfo &> /dev/null; then
    GFX_FAMILY=$(rocminfo | awk '/Name:[[:space:]]+gfx/ {print $2; exit}' | tr -d '\r[:space:]')
    echo "AMD GPU ${GFX_FAMILY} detected. Installing ROCm dependencies..."
    for c in gfx1151 gfx120x; do
        pattern=${c/%x/?}
        if [[ "$GFX_FAMILY" == $pattern ]]; then
            EXTRAS="$c"
            break
        fi
    done
    if [[ -z "$EXTRAS" ]]; then
        echo "Error: Unsupported GPU family ($GFX_FAMILY)"
        exit 1
    fi
else
    echo "No dedicated GPU detected. Installing CPU dependencies..."
    EXTRAS="cpu"
fi

uv sync --extra $EXTRAS

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate


# -----------------------------------------------------------------------------
# Tokenizer Setup (Needed for base_train to run)

if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

if ! python -c "import rustbpe" &> /dev/null; then
    uv run --no-sync --extra $EXTRAS maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    # Download the first ~2B characters of pretraining dataset
    # look at dev/repackage_data_reference.py for details on how this data was prepared
    # each data shard is ~250M chars
    # so we download 2e9 / 250e6 = 8 data shards at this point
    # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
    python -m nanochat.dataset -n 8
    # Immediately also kick off downloading more shards in the background while tokenizer trains
    # See comment below for why 240 is the right number here
    python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!
    # train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
    python -m scripts.tok_train --max_chars=2000000000
    # evaluate the tokenizer (report compression ratio etc.)
    python -m scripts.tok_eval
    
    # -----------------------------------------------------------------------------
    # Base model (pretraining)
    
    # The d20 model is 561M parameters.
    # Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
    # Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
    # At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
    # Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
    # (The total number of shards available in the entire dataset is 1822.)
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi

echo "Configuration complete. Have fun!"