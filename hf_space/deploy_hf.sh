#!/bin/bash
# Deploy the FastAPI backend to Hugging Face Spaces.
#
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login  (paste your HF token)
#
# Usage:
#   bash deploy_hf.sh

set -e

SPACE_NAME="kd-joshi/multimodal-fashion-search"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying to HuggingFace Space: $SPACE_NAME ==="

# Clone the space repo (or create it)
TEMP_DIR=$(mktemp -d)
echo "Cloning space repo into $TEMP_DIR..."

if ! git clone "https://huggingface.co/spaces/$SPACE_NAME" "$TEMP_DIR/space" 2>/dev/null; then
    echo "Space doesn't exist yet. Creating..."
    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$SPACE_NAME', repo_type='space', space_sdk='docker', private=False)
print('Space created!')
"
    git clone "https://huggingface.co/spaces/$SPACE_NAME" "$TEMP_DIR/space"
fi

cd "$TEMP_DIR/space"

# Enable Git LFS for large files
git lfs install

# Track large binary files with LFS
git lfs track "*.npy"
git lfs track "*.index"
git lfs track "*.pkl"
git lfs track "*.parquet"
git lfs track "*.png"
git lfs track "*.zip"

# Copy the HF Space README (contains the YAML config)
cp "$PROJECT_DIR/hf_space/README.md" ./README.md

# Copy the HF-specific Dockerfile
cp "$PROJECT_DIR/hf_space/Dockerfile" ./Dockerfile

# Copy application code
for f in app.py config.py similarity_search.py similarity_engine.py feature_engine.py \
         data_loader.py hybrid_search.py reranker.py build_index.py requirements.txt \
         demo.html; do
    cp "$PROJECT_DIR/$f" ./ 2>/dev/null || true
done

# Copy subdirectories
cp -r "$PROJECT_DIR/k8s" ./ 2>/dev/null || true
cp -r "$PROJECT_DIR/tests" ./ 2>/dev/null || true
cp -r "$PROJECT_DIR/assets" ./ 2>/dev/null || true

# Copy the dataset archive
mkdir -p data
cp "$PROJECT_DIR/data/archive.zip" ./data/ 2>/dev/null || true

# Copy pre-built indices (the key part — these are ~462MB)
echo "Copying pre-built indices (~462MB, this may take a moment)..."
mkdir -p indices
cp "$PROJECT_DIR/indices/"* ./indices/

# Stage everything
git add -A
git status

echo ""
echo "=== Ready to push! ==="
echo "Review the staged files above, then run:"
echo "  cd $TEMP_DIR/space && git commit -m 'Deploy multimodal fashion search' && git push"
echo ""
echo "Or to push automatically, uncomment the lines below in the script."
# git commit -m "Deploy multimodal fashion search API"
# git push
