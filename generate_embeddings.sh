# Exit immediately if a command fails, to prevent partial runs.
set -e

# --- 1. Environment Setup ---
echo "--- Setting up Python environment ---"
rm -rf .venv/
uv venv
uv add "fastembed>=0.7.1" "fastparquet>=2024.11.0" "flagembedding>=1.3.5" "numpy>=2.3.1" "polars>=1.31.0" "pylate>=1.2.0"
source .venv/bin/activate

# --- 2. Generate Embeddings ---
echo "--- Generating embeddings ---"
python scripts/generate_embeddings.py \
    --input-file data/products_text.csv \
    --text-column text \
    --output-file data/gtejina_colbert_product_embeddings.parquet \
    --model-config-path configs/supported_models.json \
    --model-name jina_colbert \
    --chunksize 10000


# --- 3. Teardown ---
echo "--- All embeddings generated. ---"
deactivate
rm -rf .venv/

echo "--- Full embedding generation finished successfully! ---"