import argparse
import os
import sys
from pathlib import Path


# Listed here, not imported, so numpy and torch are not loaded before main sets thread limits.
MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "naive_bayes",
    "gradient_boosting",
)
NEURAL_MODELS = ("mlp", "cnn", "gru", "transformer", "dnabert2", "dnabert2_scratch")


def build_parser() -> argparse.ArgumentParser:
    """Create commands for dataset preparation, steps 1–3, and task 4."""
    parser = argparse.ArgumentParser(
        description="Explore DNA/RNA as a language through project step 3, then classify it."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Download and prepare a balanced DeePaC sample."
    )
    analyze = commands.add_parser("analyze", help="Analyse an existing sequence CSV.")
    explore = commands.add_parser(
        "explore", help="Extend saved analysis without relearning word vectors."
    )
    everything = commands.add_parser(
        "all", help="Prepare DeePaC data and run steps 1–3."
    )
    for command in (prepare, everything):
        command.add_argument("--data-dir", type=Path, default=Path("data"))
        command.add_argument("--per-class", type=int, default=5000)
        command.add_argument(
            "--force",
            action="store_true",
            help="Replace the existing processed sample.",
        )
    for command in (prepare, analyze, everything):
        command.add_argument("--seed", type=int, default=42)
    for command in (analyze, explore):
        command.add_argument(
            "--input", type=Path, default=Path("data/processed/sequences.csv")
        )
    explore.add_argument("--output-dir", type=Path, default=Path("outputs"))
    analyze.add_argument("--alphabet", choices=("dna", "rna"), default="dna")
    for command in (analyze, everything):
        command.add_argument("--output-dir", type=Path, default=Path("outputs"))
        command.add_argument(
            "--k", type=int, nargs="+", default=[3, 4, 6], help="Word sizes to compare."
        )
        command.add_argument("--vector-size", type=int, default=64)
        command.add_argument("--window", type=int, default=5)
        command.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="Passes over the words during learning.",
        )
    splits = commands.add_parser(
        "prepare-splits",
        help="Download and sample the published DeePaC train, validation, and test files.",
    )
    simple = commands.add_parser(
        "simple-models",
        help="Run baselines and simple models on prepared splits (task 4).",
    )
    neural = commands.add_parser(
        "neural-models",
        help="Train small neural networks on prepared splits (task 4; needs torch).",
    )
    task4 = commands.add_parser(
        "task4", help="Prepare the splits, then run the simple models (task 4)."
    )
    for command in (splits, task4):
        command.add_argument("--data-dir", type=Path, default=Path("data"))
        command.add_argument("--train-per-class", type=int, default=50000)
        command.add_argument("--val-per-class", type=int, default=10000)
        command.add_argument("--test-per-class", type=int, default=10000)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument(
            "--force",
            action="store_true",
            help="Replace the existing processed splits.",
        )
    extra = commands.add_parser(
        "prepare-extra",
        help="Add BacPaCS and shortened-read evaluation sets to prepared splits.",
    )
    extra.add_argument("--data-dir", type=Path, default=Path("data"))
    extra.add_argument("--per-class", type=int, default=50000)
    extra.add_argument("--seed", type=int, default=42)
    genome = commands.add_parser(
        "genome-report",
        help="Add per-genome statistics to saved per-read scores of a results folder.",
    )
    genome.add_argument("--results-dir", type=Path, required=True)
    genome.add_argument("--output-dir", type=Path, default=None)
    genome.add_argument(
        "--curve-splits",
        nargs="+",
        default=["test_1", "bacpacs_test_1"],
        help="Splits for which to compute the reads-per-genome curve.",
    )
    interpret = commands.add_parser(
        "interpret",
        help="Per-genome word effects, importances and embedding checks (task 5).",
    )
    interpret.add_argument("--output-dir", type=Path, default=Path("outputs/task5"))
    interpret.add_argument("--fit-reads", type=int, default=400000)
    interpret.add_argument("--workers", type=int, default=16)
    interpret.add_argument("--seed", type=int, default=42)
    figures = commands.add_parser(
        "final-report",
        help="Rebuild comparison figures, per-model plots and the task 4 summary.",
    )
    figures.add_argument("--results-dir", type=Path, default=Path("results/task4"))
    figures.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    figures.add_argument("--task5-dir", type=Path, default=Path("outputs/task5"))
    for command in (splits, simple, neural, task4, extra, genome, interpret):
        command.add_argument(
            "--splits-dir", type=Path, default=Path("data/processed/task4")
        )
    for command in (simple, neural, task4):
        command.add_argument(
            "--train-limit",
            type=int,
            default=0,
            help="Use only the first N (shuffled) training reads; 0 keeps all.",
        )
        command.add_argument(
            "--fit-limit",
            type=int,
            default=0,
            help="Learn tf-idf weights and word vectors from the first N training reads; 0 uses all.",
        )
        command.add_argument(
            "--no-scores",
            action="store_true",
            help="Do not save per-read scores (and network weights).",
        )
    for command in (simple, task4):
        command.add_argument(
            "--output-dir", type=Path, default=Path("results/task4/simple_models")
        )
        command.add_argument(
            "--feature-dir",
            type=Path,
            default=Path("data/features/task4"),
            help="Where learned representations are cached.",
        )
        command.add_argument(
            "--k", type=int, nargs="+", default=[3, 4, 6], help="Word sizes to compare."
        )
        command.add_argument(
            "--models",
            nargs="+",
            choices=MODELS,
            default=list(MODELS),
            help="Simple models to run.",
        )
        command.add_argument("--vector-size", type=int, default=64)
        command.add_argument("--window", type=int, default=5)
        command.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="Word2Vec passes over the training reads.",
        )
        command.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=[42, 43, 44],
            help="Random settings; the first drives tuning and Word2Vec.",
        )
        command.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Word2Vec CPU workers; more than 1 loses exact repeatability.",
        )
        command.add_argument(
            "--n-jobs",
            type=int,
            default=0,
            help="CPU workers for forests and boosting; 0 uses every core.",
        )
        command.add_argument("--trees", type=int, default=200)
        command.add_argument(
            "--bootstrap",
            type=int,
            default=100,
            help="Genome-level bootstrap resamples per interval; 0 disables.",
        )
    neural.add_argument(
        "--output-dir", type=Path, default=Path("results/task4/neural_models")
    )
    neural.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features/task4"),
        help="Where learned representations are cached (shared with simple-models).",
    )
    neural.add_argument(
        "--simple-results",
        type=Path,
        default=Path("results/task4/simple_models/results_summary.csv"),
        help="Simple-model summary to compare against, if it exists.",
    )
    neural.add_argument(
        "--k", type=int, nargs="+", default=[3, 6], help="Word sizes to compare."
    )
    neural.add_argument(
        "--models",
        nargs="+",
        choices=NEURAL_MODELS,
        default=["mlp", "cnn", "gru", "transformer"],
        help="Networks to train (dnabert2 variants are opt-in).",
    )
    neural.add_argument("--vector-size", type=int, default=64)
    neural.add_argument("--window", type=int, default=5)
    neural.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Word2Vec passes over the training reads.",
    )
    neural.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random settings; the first drives tuning and Word2Vec.",
    )
    neural.add_argument("--workers", type=int, default=1)
    neural.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="CPU threads for torch; 0 uses every core.",
    )
    neural.add_argument("--bootstrap", type=int, default=100)
    neural.add_argument(
        "--max-epochs",
        type=int,
        default=15,
        help="Upper limit on passes over the training reads.",
    )
    neural.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Epochs without validation improvement before stopping.",
    )
    neural.add_argument("--batch-size", type=int, default=256)
    neural.add_argument(
        "--max-grid",
        type=int,
        default=99,
        help="Largest number of grid settings tried per model.",
    )
    neural.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    return parser


def main() -> int:
    """Run the requested command and report understandable input errors.

    Returns:
        Zero on success, or one if the command cannot complete.
    """
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= getattr(args, "seed", 42) <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295.")
    for name in (
        "per_class",
        "train_per_class",
        "val_per_class",
        "test_per_class",
        "vector_size",
        "window",
        "epochs",
        "workers",
        "trees",
        "max_epochs",
        "patience",
        "batch_size",
    ):
        if getattr(args, name, 1) < 1:
            parser.error("--" + name.replace("_", "-") + " must be at least 1.")
    if any(k < 1 for k in getattr(args, "k", [])):
        parser.error("Word sizes must be at least 1.")
    if any(not 0 <= seed <= 2**32 - 1 for seed in getattr(args, "seeds", [])):
        parser.error("--seeds must be between 0 and 4294967295.")
    if getattr(args, "bootstrap", 0) < 0:
        parser.error("--bootstrap must be at least 0.")
    threads = "1"
    if hasattr(args, "n_jobs"):
        args.n_jobs = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
        threads = str(args.n_jobs)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, threads)
    cache = Path(".cache/matplotlib").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    try:
        if args.command in ("prepare-splits", "task4"):
            from src.splits import prepare_splits

            prepare_splits(
                args.data_dir,
                args.splits_dir,
                {
                    "train": args.train_per_class,
                    "val": args.val_per_class,
                    "test_1": args.test_per_class,
                    "test_2": args.test_per_class,
                },
                args.seed,
                args.force,
            )
        if args.command == "interpret":
            from src.interpret import run_interpretation

            run_interpretation(
                args.splits_dir,
                args.output_dir,
                fit_reads=args.fit_reads,
                seed=args.seed,
                workers=args.workers,
            )
        if args.command == "final-report":
            from src.final_report import build

            build(args.results_dir, args.outputs_dir, args.task5_dir)
        if args.command == "genome-report":
            from src.genome_report import run_genome_report

            run_genome_report(
                args.splits_dir,
                args.results_dir,
                args.output_dir or args.results_dir / "genome",
                tuple(args.curve_splits),
            )
        if args.command == "prepare-extra":
            from src.splits import prepare_extra_sets

            prepare_extra_sets(
                args.data_dir, args.splits_dir, args.per_class, args.seed
            )
        if args.command in ("simple-models", "task4"):
            from src.simple_models import run_simple_models

            run_simple_models(
                splits_dir=args.splits_dir,
                output_dir=args.output_dir,
                feature_dir=args.feature_dir,
                ks=tuple(args.k),
                models=tuple(args.models),
                vector_size=args.vector_size,
                window=args.window,
                epochs=args.epochs,
                seeds=tuple(args.seeds),
                workers=args.workers,
                n_jobs=args.n_jobs,
                trees=args.trees,
                bootstrap=args.bootstrap,
                train_limit=args.train_limit,
                fit_limit=args.fit_limit,
                save_per_read=not args.no_scores,
            )
        if args.command == "neural-models":
            from src.neural_models import run_neural_models

            run_neural_models(
                splits_dir=args.splits_dir,
                output_dir=args.output_dir,
                feature_dir=args.feature_dir,
                simple_summary=args.simple_results,
                ks=tuple(args.k),
                models=tuple(args.models),
                vector_size=args.vector_size,
                window=args.window,
                epochs=args.epochs,
                seeds=tuple(args.seeds),
                workers=args.workers,
                n_jobs=args.n_jobs,
                bootstrap=args.bootstrap,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                device=args.device,
                train_limit=args.train_limit,
                fit_limit=args.fit_limit,
                max_grid=args.max_grid,
                save_per_read=not args.no_scores,
            )
        if args.command in ("prepare", "all"):
            from src.data import prepare_dataset

            input_path = prepare_dataset(
                args.data_dir, args.per_class, args.seed, args.force
            )
        elif args.command in ("analyze", "explore"):
            input_path = args.input
        if args.command in ("analyze", "all"):
            from src.analysis import analyze_dataset

            analyze_dataset(
                input_path=input_path,
                output_dir=args.output_dir,
                ks=tuple(args.k),
                alphabet=getattr(args, "alphabet", "dna"),
                vector_size=args.vector_size,
                window=args.window,
                epochs=args.epochs,
                seed=args.seed,
            )
        elif args.command == "explore":
            from src.analysis import extend_saved_analysis

            extend_saved_analysis(input_path, args.output_dir)
    except ImportError as error:
        print(
            "Missing dependency: "
            + str(error)
            + ". Run .venv/bin/python -m pip install -r "
            + (
                "requirements-neural.txt"
                if "torch" in str(error)
                else "requirements.txt"
            ),
            file=sys.stderr,
        )
        return 1
    except (ValueError, RuntimeError, OSError) as error:
        print("Could not complete the command: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
