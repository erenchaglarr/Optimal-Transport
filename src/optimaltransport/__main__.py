from __future__ import annotations

import argparse

from omegaconf import OmegaConf
from pathlib import Path
from .evaluate import (
    evaluate_checkpoint,
    evaluate_knn_on_eqx_checkpoints,
    evaluate_transportplans_with_classifier,
)
from .visualize import visualize_checkpoint
from .train import run_training_pipeline, train_image_classifier
from .sinkhorn2_eletric_bugaloo import embed_and_run_sinkhorn
from .eval_variance import (
    evaluate_transport_variance_across_dims,
    summarize_variance_results_by_dim,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml")

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train",
            "train_classifier",
            "evaluate",
            "knn",
            "visualize",
            "all",
            "sinkhorn",
            "transport_classifier",
            "variance",
        ],
        default="all",
    )
    

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
    )

    parser.add_argument("--source-class", type=int, default=5)
    parser.add_argument("--target-class", type=int, default=9)
    parser.add_argument("--max-points", type=int, default=50)
    
    parser.add_argument(
    "--latent-dims",
    nargs="+",
    type=int,
    default=[1, 2, 3, 5, 10],
    help="Latent dimensions to evaluate, e.g. --latent-dims 1 2 3 5 10 20 30 50",
)

    parser.add_argument(
    "--variance-csv",
    type=str,
    default="transport_variance_across_dims.csv",
    help="CSV file for saving transported/target variance diagnostics.",
)

    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    checkpoint_path = args.checkpoint

    if args.mode == "train_classifier":
        train_image_classifier(config)
        return

    if checkpoint_path is None:
        checkpoint_paths = [
            Path(config.paths.model_dir) / config.paths.final_model_name
        ]
    else:
        checkpoint_paths = [Path(checkpoint_path)]

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

    if args.mode == "sinkhorn":
        embed_and_run_sinkhorn(
            config=config,
            checkpoint_path=checkpoint_path,
            split=args.split,
            source_class=args.source_class,
            target_class=args.target_class,
            max_points=args.max_points,
        )

    if args.mode == "transport_classifier":
        if args.classifier_checkpoint is None:
            raise ValueError(
                "You must provide --classifier-checkpoint when using "
                "--mode transport_classifier"
            )

        evaluate_transportplans_with_classifier(
            config=config,
            classifier_checkpoint_path=args.classifier_checkpoint,
            autoencoder_checkpoint_path=checkpoint_path,
            split=args.split,
            max_points=args.max_points,
        )

    if args.mode == "knn":
        evaluate_knn_on_eqx_checkpoints(
            config=config,
            checkpoint_paths=checkpoint_paths,
            export_path=Path(config.paths.model_dir) / "knn_eqx_checkpoint_results.csv",
        )
    if args.mode == "variance":
        checkpoint_paths = {
        dim: Path(config.paths.model_dir) / f"latent{dim}.eqx"
        for dim in args.latent_dims
    }

        print("\n========== Transport variance diagnostics ==========")
        print(f"Split: {args.split}")
        print(f"Latent dimensions: {args.latent_dims}")
        print(f"Max points: {args.max_points}")
        print(f"Output CSV: {args.variance_csv}")

        variance_results = evaluate_transport_variance_across_dims(
            config=config,
            checkpoint_paths=checkpoint_paths,
            split=args.split,
            max_points=args.max_points,
            export_csv_path=args.variance_csv,
        )

        summarize_variance_results_by_dim(variance_results)



if __name__ == "__main__":
    main()
