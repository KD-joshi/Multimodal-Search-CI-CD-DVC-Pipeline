#!/bin/bash
# Deploy the FastAPI backend to Hugging Face Spaces using the Python API (Bypasses Git issues)
set -e

HF_USER=$(.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
SPACE_NAME="$HF_USER/multimodal-fashion-search"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying to HuggingFace Space: $SPACE_NAME ==="

# 1. Create a clean staging directory
TEMP_DIR=$(mktemp -d)
STAGING="$TEMP_DIR/space"
mkdir -p "$STAGING"

echo "Staging files in $STAGING..."

# 2. Copy the HF Space README (contains the YAML config) and Dockerfile
cp "$PROJECT_DIR/hf_space/README.md" "$STAGING/README.md"
cp "$PROJECT_DIR/hf_space/Dockerfile" "$STAGING/Dockerfile"

# 3. Copy application code
for f in app.py config.py similarity_search.py similarity_engine.py feature_engine.py \
         data_loader.py hybrid_search.py reranker.py build_index.py requirements.txt \
         demo.html; do
    cp "$PROJECT_DIR/$f" "$STAGING/" 2>/dev/null || true
done

# 4. Copy subdirectories
cp -r "$PROJECT_DIR/k8s" "$STAGING/" 2>/dev/null || true
cp -r "$PROJECT_DIR/tests" "$STAGING/" 2>/dev/null || true
cp -r "$PROJECT_DIR/assets" "$STAGING/" 2>/dev/null || true

# 5. Copy the dataset archive
mkdir -p "$STAGING/data"
cp "$PROJECT_DIR/data/archive.zip" "$STAGING/data/" 2>/dev/null || true

# 6. Copy pre-built indices (~462MB)
echo "Copying pre-built indices (~462MB)..."
mkdir -p "$STAGING/indices"
cp "$PROJECT_DIR/indices/"* "$STAGING/indices/"

echo "Files staged successfully. Uploading directly via HuggingFace API..."

# 7. Use the Python API to create repo and upload folder directly (bypassing Git)
.venv/bin/python -c "
from huggingface_hub import HfApi
import sys

api = HfApi()
repo_id = '$SPACE_NAME'
staging_dir = '$STAGING'

try:
    print(f'Checking if {repo_id} exists...')
    api.repo_info(repo_id, repo_type='space')
except Exception:
    print(f'Creating space {repo_id}...')
    try:
        api.create_repo(repo_id, repo_type='space', space_sdk='docker', private=False)
    except Exception as e:
        print(f'❌ Failed to create space: {e}')
        print('Make sure your HuggingFace token has "Write" permissions (specifically to create repositories).')
        sys.exit(1)

print('Uploading files... This may take a few minutes for 460MB.')
try:
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=repo_id,
        repo_type='space',
        commit_message='Deploy multimodal fashion search API'
    )
    print('✅ Upload complete! The space is now building on HuggingFace.')
except Exception as e:
    print(f'❌ Upload failed: {e}')
    sys.exit(1)
"

rm -rf "$TEMP_DIR"
