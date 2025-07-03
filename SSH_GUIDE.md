# Quick Start Guide for SSH Server Training

## 🚀 Current Status
✅ Local ViT model created successfully  
✅ Single GPU detected (Tesla V100-DGXS-32GB)  
✅ Network connectivity working  
✅ Virtual environment active  

## 📋 Quick Commands (Run in order)

### 1. Test Everything Works
```bash
chmod +x test_quick.sh start_training.sh monitor.sh
./test_quick.sh
```

### 2. Start Training
```bash
./start_training.sh
```

### 3. Monitor Training
```bash
./monitor.sh
```

## 📊 How to Check if Training is Running

### Method 1: Use the Monitor Script
```bash
./monitor.sh
```

### Method 2: Check Process
```bash
# Check if training process exists
ps aux | grep "python.*train.py"

# Check specific PID (if you have it)
ps aux | grep <PID>
```

### Method 3: Check GPU Usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Single check
nvidia-smi
```

### Method 4: Check Log Files
```bash
# Find latest training
ls -la outputs_*

# View live logs
tail -f outputs_*/training_*.log

# Check log size (should be growing)
ls -lah outputs_*/training_*.log
```

## 🔧 Common Issues & Solutions

### If Training Fails to Start:
1. Check data path: Update `DATA_DIR` in `start_training.sh`
2. Run test: `./test_quick.sh`
3. Check logs: `cat outputs_*/training_*.log`

### If Process Dies:
1. Check memory usage: `free -h`
2. Check disk space: `df -h`
3. Review error logs: `grep -i error outputs_*/training_*.log`

### If GPU Not Used:
1. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check processes: `nvidia-smi`

## 📁 Key Files Created
- `./local_vit_model/` - Pre-trained ViT model (works offline)
- `./working_model.txt` - Points to working model
- `./start_training.sh` - Simple training starter
- `./monitor.sh` - Training monitor dashboard
- `./test_quick.sh` - Quick system test

## 🎯 Training Configuration
- **Model**: Local ViT (./local_vit_model)
- **GPU**: Single Tesla V100 (32GB)
- **Batch Size**: 16
- **Epochs**: 10
- **Data**: /home/Saif/Pratham/ELC/prostate-cancer-grade-assessment

## 📈 Expected Training Signs
- Log file growing in size
- GPU utilization > 80%
- Checkpoint files being created
- Epoch progress in logs
- Memory usage stable

## ⏹️ How to Stop Training
```bash
# Method 1: Use PID from start_training.sh output
kill <PID>

# Method 2: Find and kill process
ps aux | grep "python.*train.py" | awk '{print $2}' | xargs kill

# Method 3: Kill all Python processes (careful!)
pkill -f "python.*train.py"
```

## 📞 Next Steps
1. **Run**: `./test_quick.sh` (verify everything works)
2. **Start**: `./start_training.sh` (begin training)  
3. **Monitor**: `./monitor.sh` (check progress)
4. **Wait**: Training will take several hours
5. **Results**: Check `outputs_*/checkpoints/best_model.pth`
