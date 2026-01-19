import torch
import torch.nn.functional as F
from pathlib import Path
import transformer_flow as tf


def interpolate_spatial(tensor, old_patch=8, new_patch=4, channels=3):
    """Bicubic interpolate spatial weights from old_patch to new_patch size."""
    spatial = tensor.reshape(-1, channels, old_patch, old_patch).float()
    resized = F.interpolate(spatial, size=(new_patch, new_patch), mode='bicubic', align_corners=False)
    return resized.reshape(tensor.shape[0], -1).to(tensor.dtype)


def _load_state_dict_custom(model, ckpt, old_patch=8, new_patch=4, in_channels=3):
    """
    Load AFHQ checkpoint into ImageNet model with bicubic interpolation
    for spatial weights and random init for class embeddings.
    """
    filtered_dict = {}
    skipped_keys = []
    interpolated_keys = []
    
    for k, v in ckpt.items():
        # Skip class_embed - let model use random initialization
        if 'class_embed' in k:
            skipped_keys.append(k)
            continue
        
        # Interpolate var buffer: [num_patches, in_channels * old_patch^2] -> [num_patches, in_channels * new_patch^2]
        if k == 'var':
            # [1024, 192] -> [1024, 48]
            interpolated = interpolate_spatial(v, old_patch, new_patch, in_channels)
            filtered_dict[k] = interpolated
            interpolated_keys.append(f"{k}: {list(v.shape)} -> {list(interpolated.shape)}")
            continue
        
        # Interpolate proj_in.weight: [channels, in_channels * old_patch^2] -> [channels, in_channels * new_patch^2]
        if 'proj_in.weight' in k:
            # [768, 192] -> [768, 48]
            interpolated = interpolate_spatial(v, old_patch, new_patch, in_channels)
            filtered_dict[k] = interpolated
            interpolated_keys.append(f"{k}: {list(v.shape)} -> {list(interpolated.shape)}")
            continue
        
        # Interpolate proj_out.weight: [out_channels, channels] where out_channels = 2 * in_channels * patch^2 (NVP)
        if 'proj_out.weight' in k:
            # [384, 768] -> [96, 768]
            # Transpose to [768, 384], reshape to [768, 6, 8, 8], interpolate, reshape back
            nvp_channels = 2 * in_channels  # 6 for RGB with NVP
            v_t = v.t()  # [768, 384]
            spatial = v_t.reshape(v_t.shape[0], nvp_channels, old_patch, old_patch).float()
            resized = F.interpolate(spatial, size=(new_patch, new_patch), mode='bicubic', align_corners=False)
            interpolated = resized.reshape(v_t.shape[0], -1).t().to(v.dtype)  # [96, 768]
            filtered_dict[k] = interpolated
            interpolated_keys.append(f"{k}: {list(v.shape)} -> {list(interpolated.shape)}")
            continue
        
        # Interpolate proj_out.bias: [out_channels] where out_channels = 2 * in_channels * patch^2 (NVP)
        if 'proj_out.bias' in k:
            # [384] -> [96]
            nvp_channels = 2 * in_channels  # 6 for RGB with NVP
            spatial = v.reshape(1, nvp_channels, old_patch, old_patch).float()
            resized = F.interpolate(spatial, size=(new_patch, new_patch), mode='bicubic', align_corners=False)
            interpolated = resized.reshape(-1).to(v.dtype)  # [96]
            filtered_dict[k] = interpolated
            interpolated_keys.append(f"{k}: {list(v.shape)} -> {list(interpolated.shape)}")
            continue
        
        # All other weights (attention blocks, etc.) copy directly
        filtered_dict[k] = v
    
    print(f"Skipped keys (random init): {skipped_keys}")
    print(f"Interpolated keys:")
    for ik in interpolated_keys:
        print(f"  {ik}")
    
    msg = model.load_state_dict(filtered_dict, strict=False)
    print(f"\nLoad Result: {msg}")
    return model


def main():
    afhq_ckpt = Path("models/afhq_model_8_768_8_8_0.07.pth")
    imagenet_ckpt = Path("models/imagenet_model_converted.pth")

    settings = {
        "img_size": 128,
        "channel_size": 3,
        "patch_size": 4,
        "channels": 768,
        "blocks": 8,
        "layers_per_block": 8,
        "nvp": True,
        "num_classes": 1000,
    }

    model = tf.Model(
        in_channels=settings["channel_size"],
        img_size=settings["img_size"],
        patch_size=settings["patch_size"],
        channels=settings["channels"],
        num_blocks=settings["blocks"],
        layers_per_block=settings["layers_per_block"],
        nvp=settings["nvp"],
        num_classes=settings["num_classes"],
    )
    
    model = _load_state_dict_custom(
        model, 
        torch.load(afhq_ckpt, map_location='cpu', weights_only=True),
        old_patch=8,
        new_patch=settings["patch_size"],
        in_channels=settings["channel_size"],
    )

    torch.save(model.state_dict(), imagenet_ckpt)
    print(f"\nSaved converted model to {imagenet_ckpt}")


if __name__ == "__main__":
    main()
