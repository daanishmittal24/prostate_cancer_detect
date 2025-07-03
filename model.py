import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTForProstateCancer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pre-trained ViT with error handling
        try:
            print(f"Loading ViT model: {config.model_name}")
            self.vit = ViTModel.from_pretrained(config.model_name)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {config.model_name}: {e}")
            print("🔄 Trying alternative model...")
            try:
                # Fallback to a different model
                fallback_model = "google/vit-base-patch16-224"
                print(f"Trying fallback model: {fallback_model}")
                self.vit = ViTModel.from_pretrained(fallback_model)
                print("✅ Fallback model loaded successfully")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                print("🔄 Creating model from config...")
                # Last resort: create from config
                vit_config = ViTConfig(
                    image_size=config.img_size,
                    patch_size=16,
                    num_channels=3,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.0,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    layer_norm_eps=1e-12,
                    num_labels=config.num_classes
                )
                self.vit = ViTModel(vit_config)
                print("✅ Model created from scratch")
        
        # Freeze the base model if needed
        # for param in self.vit.parameters():
        #     param.requires_grad = False
        
        # Get the hidden size from ViT
        hidden_size = self.vit.config.hidden_size
        
        # Store device information
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        self.use_dp = self.n_gpu > 1 and not config.distributed
        self.use_ddp = config.distributed and self.n_gpu > 1
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, config.num_classes)
        )
        
        # Segmentation head (Upsampling path)
        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, pixel_values):
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values)
        
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Classification: Use the [CLS] token for classification
        cls_token = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        logits = self.classifier(cls_token)  # (batch_size, num_classes)
        
        # Segmentation: Reshape and upsample
        # Get the patch embeddings (excluding CLS token)
        patch_embeddings = last_hidden_state[:, 1:, :]  # (batch_size, num_patches, hidden_size)
        
        # Reshape to 2D feature map
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        height = width = int(num_patches ** 0.5)
        
        # Reshape to (batch_size, hidden_size, height, width)
        features = patch_embeddings.permute(0, 2, 1).view(batch_size, hidden_size, height, width)
        
        # Get segmentation mask
        mask = self.upsample(features)  # (batch_size, 1, img_size, img_size)
        
        return logits, mask

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, pred_logits, pred_masks, targets, masks=None):
        # Classification loss
        cls_loss = self.ce_loss(pred_logits, targets)
        
        # Only compute segmentation loss if masks are provided
        seg_loss = 0
        if masks is not None:
            seg_loss = self.bce_loss(pred_masks, masks.unsqueeze(1))
        
        # Combined loss
        total_loss = (1 - self.alpha) * cls_loss + self.alpha * seg_loss
        
        return total_loss, cls_loss, seg_loss
