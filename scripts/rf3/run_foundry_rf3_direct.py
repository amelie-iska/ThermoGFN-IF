#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from rf3.inference_engines.rf3 import RF3InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Foundry RF3 inference through the RF3InferenceEngine API."
    )
    parser.add_argument("--inputs", required=True, help="Input JSON/CIF path.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--devices-per-node", type=int, default=1, help="Number of devices per node."
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--n-recycles", type=int, default=10, help="Recycle count.")
    parser.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=5,
        help="Diffusion batch size.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=50, help="Diffusion timesteps."
    )
    parser.add_argument(
        "--local-msa-dirs",
        default="",
        help="Optional LOCAL_MSA_DIRS override.",
    )
    parser.add_argument(
        "--compress-outputs",
        action="store_true",
        help="Write compressed CIF outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose engine logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.local_msa_dirs:
        os.environ["LOCAL_MSA_DIRS"] = args.local_msa_dirs

    # This improves throughput on L40S without touching numerical mode elsewhere.
    torch.set_float32_matmul_precision("high")

    engine = RF3InferenceEngine(
        ckpt_path=args.ckpt_path,
        devices_per_node=args.devices_per_node,
        num_nodes=args.num_nodes,
        n_recycles=args.n_recycles,
        diffusion_batch_size=args.diffusion_batch_size,
        num_steps=args.num_steps,
        compress_outputs=args.compress_outputs,
        verbose=args.verbose,
    )
    engine.initialize()
    engine.run(inputs=Path(args.inputs), out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
