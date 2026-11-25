# DNA as a Language

Predicting bacterial source-organism pathogenic potential from DNA reads. We compare word counts, TF-IDF and Word2Vec with conventional classifiers, neural networks and DNABERT-2. Evaluation covers held-out genomes, sequence perturbations, BacPaCS transfer and LLM-generated explanations.

Data: [DeePaC bacterial reads](https://zenodo.org/records/3678563), with species separated across training, validation and test partitions.

## Setup

Use Python 3.9 to 3.12 for the core pipeline.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

For neural models, also install:

```bash
.venv/bin/python -m pip install -r requirements-neural.txt
```

Exploration and conventional models run on CPU. A GPU is recommended for full neural training. DNABERT-2 also needs `transformers==4.46.3` and the adapted model files at `data/models/dnabert2`, which are not included in the repository.

## Run

Explore word representations on a balanced sample of 10,000 reads:

```bash
.venv/bin/python run.py all --output-dir outputs/exploration
```

Prepare classification data with 2 million training reads, then train conventional models and the CNN:

```bash
.venv/bin/python run.py prepare-splits --train-per-class 1000000 --val-per-class 50000 --test-per-class 100000
.venv/bin/python run.py prepare-extra
.venv/bin/python run.py simple-models --train-limit 400000 --fit-limit 400000 --trees 100 --workers 8
.venv/bin/python run.py neural-models --models cnn --k 6 --seeds 42 43 --batch-size 512 --output-dir results/task4/full_cnn
.venv/bin/python run.py genome-report --results-dir results/task4/simple_models
.venv/bin/python run.py genome-report --results-dir results/task4/full_cnn
.venv/bin/python run.py interpret
.venv/bin/python run.py final-report
```

Downloads are automatic. Other neural model names are `mlp`, `gru`, `transformer`, `dnabert2` and `dnabert2_scratch`. Full run settings are recorded in `outputs/task4/full_<model>/summary.json`; use `run.py <command> --help` for options. Neural results can vary across runs and hardware.

Saved LLM prompts and replies are in `outputs/task6/`. Repeating that experiment through `src.llm_experiment.run_experiment` requires an OpenRouter API key supplied through `OPENROUTER_API_KEY` or `~/.config/csf407/openrouter_key`.

## Files

| Location | Contents |
| --- | --- |
| `run.py` | Command-line entry point |
| `src/` | Data preparation, representations, models and evaluation |
| `tests/` | Automated tests |
| `old_outputs/` | Original exploratory analysis |
| `outputs/task4/` | Classification results, run settings and comparison figures |
| `outputs/task5/` | Genome-level word effects and representation checks |
| `outputs/task6/` | LLM prompts, replies and qualitative assessment |

See [the results table](outputs/task4/summary/all_models_by_split.csv) for all evaluation conditions. Downloaded data, cached features and model weights are excluded from Git.

## Checks

```bash
.venv/bin/python -m pytest
```
