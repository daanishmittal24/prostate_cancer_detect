# Prostate Cancer Detection - Server Setup Guide

This guide explains how to set up and run the prostate cancer detection model on a multi-GPU Linux server.

## Prerequisites

1. Linux server with NVIDIA GPUs
2. CUDA and cuDNN installed
3. Python 3.7+
4. PyTorch with CUDA support
5. Other Python dependencies from `requirements.txt`

## Setup Instructions

1. **Clone the repository** to your server:
   ```bash
   git clone <repository-url>
   cd prostate-cancer-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Organize your data in the following structure:
     ```
     data/
     ├── train_images/       # Training images (.tiff)
     ├── train_label_masks/  # Training masks (.tiff)
     ├── train.csv           # Training metadata
     └── test.csv            # Test metadata
     ```
   - Update the `DATA_DIR` path in `run_on_server.sh` to point to your data directory

## Running the Training

### Quick Start

To start training with default settings:

```bash
# Make the script executable
chmod +x run_on_server.sh

# Start training
./run_on_server.sh
```

### Custom Configuration

You can modify the following parameters in `run_on_server.sh`:

- `NUM_GPUS`: Number of GPUs to use (0 = use all available)
- `BATCH_SIZE`: Batch size per GPU
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Initial learning rate
- `MODEL_NAME`: Pretrained model to use
- `OUTPUT_DIR`: Directory to save outputs (default: timestamped directory)

### Monitoring Training

1. **View logs**:
   ```bash
   tail -f outputs_<timestamp>/training_<timestamp>.log
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **View TensorBoard logs** (if enabled):
   ```bash
   tensorboard --logdir=outputs_<timestamp>/tensorboard
   ```

## Output Files

The training script will create the following directory structure:

```
outputs_<timestamp>/
├── checkpoints/            # Model checkpoints
│   ├── model_epoch_001.pth
│   ├── model_epoch_002.pth
│   └── best_model.pth
├── metrics_epoch_001.png   # Training metrics plots
├── metrics_epoch_002.png
├── training_metrics.json   # Training metrics in JSON format
└── training_<timestamp>.log  # Training logs
```

## Stopping Training

To stop the training process:

1. Find the process ID:
   ```bash
   ps aux | grep "python -m torch.distributed.launch"
   ```

2. Kill the process:
   ```bash
   kill <process_id>
   ```

## Resuming Training

To resume training from a checkpoint:

1. Update the `--resume` parameter in `run_on_server.sh` to point to your checkpoint file:
   ```bash
   --resume outputs_<timestamp>/checkpoints/model_epoch_010.pth
   ```

2. Run the training script again.

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE` in `run_on_server.sh`
   - Use gradient accumulation
   - Use mixed precision training

2. **NCCL Errors**:
   - Make sure all GPUs are properly detected
   - Try setting `NCCL_DEBUG=INFO` for more detailed error messages

3. **Slow Training**:
   - Increase `num_workers` in the DataLoader
   - Use a faster storage solution (e.g., SSD)
   - Enable mixed precision training

## Performance Tips

1. **Mixed Precision Training**:
   Install Apex and enable mixed precision training for faster training and reduced memory usage.

2. **Gradient Accumulation**:
   For very large models, use gradient accumulation to effectively increase batch size.

3. **Data Loading**:
   - Use `pin_memory=True` in DataLoader
   - Use multiple workers for data loading
   - Consider using NVIDIA DALI for faster data loading

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
