import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create commands for dataset preparation and steps 1–3."""
    parser = argparse.ArgumentParser(
        description="Explore DNA/RNA as a language through project step 3."
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
    for name in ("per_class", "vector_size", "window", "epochs"):
        if getattr(args, name, 1) < 1:
            parser.error("--" + name.replace("_", "-") + " must be at least 1.")
    if any(k < 1 for k in getattr(args, "k", [])):
        parser.error("Word sizes must be at least 1.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, "1")
    cache = Path(".cache/matplotlib").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    try:
        if args.command in ("prepare", "all"):
            from src.data import prepare_dataset

            input_path = prepare_dataset(
                args.data_dir, args.per_class, args.seed, args.force
            )
        else:
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
            + ". Run .venv/bin/python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except (ValueError, RuntimeError, OSError) as error:
        print("Could not complete the command: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
