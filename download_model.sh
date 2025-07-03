#!/bin/bash

# Pre-download Model Script
echo "=== Pre-downloading ViT Model ==="
echo "This script downloads the ViT model before training to avoid network issues during training"
echo

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "1. Testing network connectivity..."
python -c "
import requests
try:
    response = requests.get('https://huggingface.co/', timeout=10)
    if response.status_code == 200:
        print('✅ Network connectivity OK')
    else:
        print('⚠️  Network issues detected')
except Exception as e:
    print(f'❌ Network error: {e}')
"

echo
echo "2. Downloading ViT model..."

python -c "
import os
from transformers import ViTModel, ViTConfig

models_to_try = [
    'google/vit-base-patch16-224-in21k',
    'google/vit-base-patch16-224',
    'facebook/deit-base-patch16-224'
]

for model_name in models_to_try:
    try:
        print(f'Trying to download: {model_name}')
        model = ViTModel.from_pretrained(model_name)
        print(f'✅ Successfully downloaded: {model_name}')
        
        # Save the working model name to a file
        with open('working_model.txt', 'w') as f:
            f.write(model_name)
        print(f'✅ Saved working model name to working_model.txt')
        break
        
    except Exception as e:
        print(f'❌ Failed to download {model_name}: {e}')
        continue
else:
    print('❌ All model downloads failed')
    print('Creating basic ViT config for local use...')
    
    # Create a basic config
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12
    )
    
    model = ViTModel(config)
    
    # Save locally
    os.makedirs('local_vit_model', exist_ok=True)
    model.save_pretrained('local_vit_model')
    config.save_pretrained('local_vit_model')
    
    with open('working_model.txt', 'w') as f:
        f.write('./local_vit_model')
    
    print('✅ Created local ViT model')
"

echo
echo "3. Verifying model..."
if [ -f "working_model.txt" ]; then
    WORKING_MODEL=$(cat working_model.txt)
    echo "Working model: $WORKING_MODEL"
    
    python -c "
from transformers import ViTModel
import torch

model_name = '$WORKING_MODEL'
try:
    model = ViTModel.from_pretrained(model_name)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print('✅ Model verification successful')
    print(f'   Model output shape: {output.last_hidden_state.shape}')
except Exception as e:
    print(f'❌ Model verification failed: {e}')
"
else
    echo "❌ No working model found"
fi

echo
echo "Done! Now you can run training with the pre-downloaded model."
