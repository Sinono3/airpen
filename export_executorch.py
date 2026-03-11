import argparse
from pathlib import Path

import torch
from torch.export import export

from net import Model  # <- your Model in net.py

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="Path to model.pth")
    ap.add_argument("--out", type=Path, required=True, help="Output .pte path")
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument(
        "--time",
        type=int,
        default=132,
        help="Sequence length (time dimension). Must match what you'll run on-device.",
    )
    args = ap.parse_args()

    # Always export on CPU
    device = torch.device("cpu")

    # 1) Build your model
    model = Model(in_channels=args.in_channels, num_classes=args.num_classes).to(
        device
    )

    # 2) Load your state_dict (your training code saves state_dict)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3) Example input (N, C, T) to match your pipeline: (B, 3, 132) etc.
    example_inputs = (
        torch.randn(args.batch, args.in_channels, args.time, device=device),
    )

    # 4) Export with torch.export
    ep = export(model, example_inputs)

    # 5) Lower for edge + partition to XNNPACK (good default for ARM CPU)
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

    # 6) Serialize ExecuTorch program
    et_program = edge.to_executorch()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        et_program.write_to_file(f)

    print(f"Exported ExecuTorch program: {args.out}")


if __name__ == "__main__":
    main()
