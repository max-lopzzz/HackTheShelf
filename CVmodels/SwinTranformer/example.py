import torch
from model import SwinTransformer

# Create Swin Transformer model
model = SwinTransformer(
    img_size=224,
    patch_size=4,
    in_channels=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.
)

# Sample input
x = torch.rand(1, 3, 224, 224)
output = model(x)
print(output.shape)  # torch.Size([1, 1000])