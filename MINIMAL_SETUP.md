# Minimal Setup Instructions for SSH Server

## What You Have Now (Essential Files Only):

### Core Training Files:
- `config.py` - Configuration settings
- `model.py` - ViT model definition  
- `data_loader.py` - Data loading functions
- `train_utils.py` - Training utilities
- `train.py` - Main training script (Python 3.6 compatible, no f-strings)
- `train_minimal.py` - Simple backup training script
- `requirements.txt` - Python dependencies

### Startup Scripts:
- `start_final.sh` - Main startup script (uses fixed train.py)
- `start_minimal.sh` - Backup startup script (uses train_minimal.py)

### Monitoring:
- `monitor_minimal.sh` - Simple monitoring script

## Manual Steps to Run on SSH Server:

### 1. Upload Essential Files to Server:
```bash
# Only upload these files to your server:
scp config.py model.py data_loader.py train_utils.py train.py requirements.txt start_final.sh monitor_minimal.sh user@server:/path/to/project/
```

### 2. On SSH Server - Setup Environment:
```bash
# Make scripts executable
chmod +x start_final.sh monitor_minimal.sh

# Install dependencies (if needed)
pip3 install -r requirements.txt
```

### 3. Update Data Path (if different):
```bash
# Edit start_final.sh and update DATA_DIR if your path is different
nano start_final.sh
# Change this line: DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
```

### 4. Start Training:
```bash
# Start training
./start_final.sh
```

### 5. Monitor Training:
```bash
# Monitor progress
./monitor_minimal.sh

# Or view live logs
tail -f outputs_final_*/training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### 6. Stop Training (if needed):
```bash
# Find and kill training process
pkill -f train.py
# Or use PID from monitor script
```

## Key Benefits of This Minimal Setup:

✅ **Python 3.6 Compatible** - No f-strings, works on older servers
✅ **Simple Dependencies** - Only essential packages needed  
✅ **Error Handling** - Robust data loading and model initialization
✅ **Easy Monitoring** - Simple scripts to check progress
✅ **Clean Structure** - Minimal files, easy to understand

## Troubleshooting:

### If training fails:
1. Check log file: `tail outputs_*/training.log`
2. Verify data path exists: `ls -la /home/Saif/Pratham/ELC/prostate-cancer-grade-assessment`
3. Check GPU: `nvidia-smi`
4. Use backup script: `./start_minimal.sh`

### If you get import errors:
```bash
# Install missing packages
pip3 install torch torchvision transformers pandas numpy scikit-learn matplotlib tqdm opencv-python-headless pillow
```

### If data path is wrong:
```bash
# Find correct path
find /home -name "*prostate*" -type d 2>/dev/null
# Update DATA_DIR in start_final.sh
```

## Summary:
Just upload the 8 essential files, run `./start_final.sh`, and monitor with `./monitor_minimal.sh`. The setup is now minimal and robust!
