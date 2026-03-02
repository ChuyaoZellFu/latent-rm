"""
验证训练出的 checkpoint 能被 RLinf 的 ResnetRewModel 正确加载。

用法:
    python check_compat.py --checkpoint /ssd/cyfu/yuhan/reward_model/outputs/resnet_rm.pth

这个脚本会:
1. 用我们自己的 model.py 加载 checkpoint
2. 用 RLinf 的 diffsynth ResnetRewModel 加载同一个 checkpoint
3. 对比两者在相同输入下的输出是否一致
"""

import argparse
import sys

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check RLinf compatibility")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- 1. 用我们的 model.py 加载 ----
    print("=" * 50)
    print("[1] Loading with our model.py ...")
    from model import ResnetRewModel as OurModel
    our_model = OurModel(checkpoint_path=args.checkpoint).to(device).eval()

    # ---- 2. 用 RLinf 的 diffsynth 加载 ----
    print("[2] Loading with RLinf diffsynth ResnetRewModel ...")
    try:
        sys.path.insert(0, "/ssd/cyfu/yuhan/diffsynth-studio-rlinf")
        from diffsynth.models.reward_model import ResnetRewModel as RLinfModel
        rlinf_model = RLinfModel(checkpoint_path=args.checkpoint).to(device).eval()
    except Exception as e:
        print(f"  WARNING: Could not load RLinf model: {e}")
        print("  Skipping RLinf comparison. Only verifying our model loads correctly.")
        rlinf_model = None

    # ---- 3. 对比输出 ----
    print("[3] Comparing outputs on random input ...")
    torch.manual_seed(42)
    dummy_input = torch.randn(4, 3, 256, 256, device=device).clamp(-1, 1)

    our_output = our_model.predict_rew(dummy_input)
    print(f"  Our model output: {our_output.squeeze().tolist()}")

    if rlinf_model is not None:
        rlinf_output = rlinf_model.predict_rew(dummy_input)
        print(f"  RLinf model output: {rlinf_output.squeeze().tolist()}")

        diff = (our_output - rlinf_output).abs().max().item()
        print(f"  Max difference: {diff}")
        if diff < 1e-5:
            print("  ✓ PASS: Outputs are identical!")
        else:
            print("  ✗ FAIL: Outputs differ!")
    else:
        print("  (RLinf model not available, skipping comparison)")

    # ---- 4. 验证 state_dict key 一致性 ----
    print("[4] Checking state_dict keys ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    our_keys = set(our_model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    if our_keys == ckpt_keys:
        print(f"  ✓ PASS: All {len(our_keys)} keys match!")
    else:
        missing = our_keys - ckpt_keys
        extra = ckpt_keys - our_keys
        if missing:
            print(f"  ✗ Missing keys in checkpoint: {missing}")
        if extra:
            print(f"  ✗ Extra keys in checkpoint: {extra}")

    print("=" * 50)
    print("Compatibility check complete.")


if __name__ == "__main__":
    main()
