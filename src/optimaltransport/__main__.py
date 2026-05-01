from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from .evaluate import evaluate_checkpoint
from .visualize import visualize_checkpoint
from .train import run_training_pipeline
from .sinkhorn2_eletric_bugaloo import cost_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "visualize", "all" , "hehe"],
        default="all",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    checkpoint_path = args.checkpoint

    if args.mode in {"train", "all"}:
        results = run_training_pipeline(config)
        checkpoint_path = results["final_results"]["final_checkpoint_path"]

    if args.mode in {"evaluate", "all"}:
        evaluate_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            split=args.split,
        )

    if args.mode in {"visualize", "all"}:
        vis_split = "train" if args.mode == "all" else args.split
        visualize_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            split=vis_split,
        )
    if args.mode in {"hehe"}:
        cost_matrix(
            config=config,
            checkpoint_path=checkpoint_path,
            split= "train",
        )




if __name__ == "__main__":
    main()