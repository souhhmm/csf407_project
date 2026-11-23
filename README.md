# DNA/RNA as a language

This project implements **steps 1–3** of the course project: turning sequences into words, exploring those words, and applying language-processing ideas. It stops before training models that predict disease classes.

The default dataset is a balanced sample of 10,000 DNA sequences from [DeePaC](https://zenodo.org/records/3678563): 5,000 from each source class. Everything runs on the CPU; a separate GPU is not needed.

## Run the project

Use Python 3.9–3.12. From the project directory, create the environment, install the recorded dependencies, and run the pipeline:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run.py all
```

This prepares the dataset if needed, then compares words of 3, 4, and 6 letters. A checked sample with the same settings is reused. Results go to `outputs/`.

Open **`outputs/report.md`** for the results, explanations, and plots. The default run produces twelve plots with white backgrounds, a consistent blue, orange and teal palette, and short labels. Bars and histogram areas use line patterns with light fills. Detailed comparisons are also saved as tables.

`requirements.txt` records the main packages and compatibility pins. `requirements-lock.txt` records every installed dependency's version. The chosen versions were checked on this Mac with Python 3.9.6.

After the first setup, rerun with `.venv/bin/python run.py all`. The first run downloads about 202 MB of source data. The local environment and downloaded data are excluded from Git. Results in `outputs/` can be committed and can also be recreated using these commands.

## Run the stages separately

Prepare the default dataset:

```bash
.venv/bin/python run.py prepare
```

Run analysis on the prepared data:

```bash
.venv/bin/python run.py analyze
```

To add or refresh the deeper comparisons using matching saved results, without relearning word vectors:

```bash
.venv/bin/python run.py explore
```

This command checks that the input file matches the original analysis. The regular `analyze` and `all` commands also include these comparisons.

Change word sizes or how long word vectors are learned:

```bash
.venv/bin/python run.py analyze --k 3 4 6 --epochs 5 --vector-size 32 --output-dir outputs_short
```

To replace the processed sample with a smaller one, then analyse it:

```bash
.venv/bin/python run.py prepare --per-class 1000 --force
.venv/bin/python run.py analyze --output-dir outputs_small
```

`--force` replaces `data/processed/sequences.csv`. Original downloaded files are reused if they match the published fingerprints. Running the default `all` command after changing sample settings requires `--force` to restore the default sample.

Use `run.py prepare --help` or `run.py analyze --help` for all options. Running analysis again replaces results in the chosen output directory. Use a different directory to keep multiple experiments.

## What is implemented

### Step 1: turn sequences into words

Words are overlapping groups of letters. For example, `ATGCA` becomes `ATG`, `TGC`, and `GCA` with groups of three. The default comparison uses groups of 3, 4, and 6 letters.

Letter case and spacing are normalized. DNA uses `ACGT`, and RNA uses `ACGU`. T and U are not silently exchanged. Recognized unknown-letter codes are retained; words containing them are skipped. The words on opposite sides of an unknown letter stay in separate groups during context learning.

### Step 2: explore the words

The project calculates sequence lengths, letter proportions, word counts, common words, rare observed words, and how much of the possible vocabulary each sequence contains. It saves summaries for each class and compares the different word sizes.

The charts show whether sequences appear to group together after their representations are reduced to two coordinates. The original representations and the chart coordinates are both saved.

### Step 3: represent sequences using language-processing ideas

- **Bag-of-Words:** count each word in a sequence.
- **TF-IDF:** give more weight to words found in fewer sequences, then scale the sequence's weights to a comparable size.
- **Word2Vec:** learn a list of numbers for each word from its neighbours, without using class labels.
- **Whole-sequence vectors:** average learned word vectors, either counting each occurrence equally or weighting them by TF-IDF.

The project lists nearby words in the learned representation and compares word pairs' observed surroundings with their learned similarity. It includes a shuffled comparison and a second check using neighbours that share no positions in the original sequence.

### Further descriptive analysis

- **Frequency concentration:** how much of the total word count comes from the most frequent words, compared across word sizes.
- **Class word differences:** relative word rates in the two source classes. The figure uses four-letter words when available and shows up to six words in each direction, with at least 20 occurrences per class. These are descriptive differences, not disease predictors.
- **Letter composition and learned representations:** the existing sequence coordinates coloured by G and C proportions, with rank correlations for both coordinates in all four representations. Sequences with no usable words are excluded.
- **Letter order:** original word counts compared with ten shuffled copies of the dataset. Each shuffle rearranges letters within known-letter runs, preserving their letter counts, lengths and unknown gaps. The comparison uses four-letter words when available, otherwise the closest completed size up to six. No new word-vector model is trained for this check.

The class and letter-order charts show selected differences. Their complete tables are saved. Ratios use a 0.5-count adjustment to avoid division by zero. The shuffle range describes variation across these ten runs and is not a confidence interval. These checks remain within steps 1–3.

Word-vector learning uses one CPU worker, a fixed random setting, and a stable word hash. The same input and software versions produce repeatable results on this machine. Different machines or numerical libraries may produce small differences.

## Data preparation

The two compressed source files occupy about 202 MB together. Preparation reads every record once and keeps a random candidate sample of twice the requested size per class. It removes repeated candidates and sequences with conflicting labels, then selects equal final class sizes. It does not load the entire dataset into memory. If too few unique candidates remain, it reports the problem instead of silently changing the requested sample size.

The sample is balanced by class, not by bacterial species. Repeated sequences are checked within sampled candidates, not across all source genomes. Source headers and filenames are preserved. The manifest records fingerprints, selection settings, class counts, and rejected records.

Read [data/SOURCES.md](data/SOURCES.md) for the source files and citation.

## Use another DNA or RNA dataset

Provide a CSV with `id`, `sequence`, and `label` columns:

```csv
id,sequence,label
sample_1,ATGCATGC,group_a
sample_2,GGCCGGCC,group_b
```

Then run:

```bash
.venv/bin/python run.py analyze --input path/to/sequences.csv --output-dir outputs_custom
```

For RNA, also pass `--alphabet rna`. Missing values, unexpected letters, repeated IDs with different content, and conflicting labels are reported. Repeated sequences with the same label are removed and counted. Sequences too short to form a word have zero vectors for that word size. A word size with no usable words anywhere is skipped and explained in the report.

## Find the results

| Location | Contents |
| --- | --- |
| `data/processed/sequences.csv` | Prepared sequences and their original source information |
| `data/processed/manifest.json` | Data source, fingerprints, selection settings, and checks |
| `outputs/report.md` | Plain-language results with twelve minimal figures for the default dataset |
| `outputs/summary.json` | Settings, software versions, context comparisons, and runtime |
| `outputs/sequence_statistics.csv` | Lengths and letter proportions for each sequence |
| `outputs/class_statistics.csv` | Sequence summaries by class |
| `outputs/k_comparison.csv` | Comparison of word sizes |
| `outputs/frequency_concentration.csv` | Occurrence share contributed by the most frequent 10% of words |
| `outputs/composition_correlations.csv` | Agreement between G/C proportions and saved sequence coordinates |
| `outputs/k_3/`, `k_4/`, `k_6/` | Word tables, numeric representations, coordinates, and learned word-vector models |

Within each word-size directory, `row_ids.csv` gives the sequence order, and `vocabulary.json` gives the word order. `counts.npz` and `tfidf.npz` store word-based sequence representations without wasting space on zeros. `word_vectors.npy` stores the learned word vectors; `mean_vectors.npy` and `weighted_vectors.npy` store whole-sequence vectors. `idf.npy` stores the learned distinctive-word weights.

The two `context_pairs_*.csv` files contain the surroundings comparison. `nearest_words.csv` includes learned similarities and the fraction of matching letters. `common_words.csv` and `rare_words.csv` contain the highest and lowest observed counts. All observed words appear in `word_frequencies.csv`.

Each word-size directory also includes `frequency_ranks.csv`. The chosen comparison directory includes `class_word_comparison.csv` and `shuffled_word_comparison.csv` when those comparisons are available. The `*_words_shown.csv` files identify the exact words in the two ratio figures.

## Reading the results

DeePaC contains simulated short sequences from real bacterial genomes. Its labels describe the source bacteria; a labelled fragment does not necessarily cause disease itself. Differences can reflect species or letter proportions. Overlapping words also share letters by construction, so word similarity alone does not prove biological understanding.

The charts show only part of the variation in each representation. The percentages in their titles describe the part shown. Apparent groups are exploratory observations, not measured disease-prediction performance. The numeric context comparisons are also exploratory; a single shuffled comparison is not a statistical significance test.

The published validation files are used for this work. The published test files remain untouched. If the project later expands to step 4, use the published training data for learning and untouched test data for final evaluation.

## Check the code

```bash
.venv/bin/python -m pytest
```

The checks cover word creation, unknown-letter handling, data verification and selection, duplicate handling, hand-calculated word counts and TF-IDF values, vector averages, repeatability, and a complete small pipeline run. Test sequences are only test fixtures; the main report uses the downloaded DeePaC data.

To format the Python code with Black:

```bash
.venv/bin/python -m black --workers 1 run.py src tests
```

---

# Task 4: classification models

Steps 1–3 above stay unchanged. Task 4 trains classifiers to separate pathogenic from nonpathogenic source organisms. This section covers the **simple models stage**; neural models and transformers will be added to the same commands later.

## Data and evaluation design

Task 4 does not reuse the 10,000-read validation sample from steps 1–3. It samples the official DeePaC files, which the authors split **by genome**: the training, validation, and test genomes never overlap.

| Split | Source file | Used for |
| --- | --- | --- |
| `train` | `*_train.fasta.gz` | Learning word weights, Word2Vec, and classifiers |
| `val` | `*_val.fasta.gz` | Choosing hyperparameters; ranking models |
| `test_1`, `test_2` | `*_test_1.fasta.gz`, `*_test_2.fasta.gz` | Final scores for the setting validation chose |

- Reads are sampled at random from the whole of each file (the files are ordered genome by genome), balanced by class, and de-duplicated. Each read keeps its genome accession in the `genome` column.
- `prepare-splits` stops with an error if any two of train, val, and test share a genome. Exact repeated reads across splits are counted in `data/processed/task4/manifest.json`.
- `test_1` and `test_2` are the two mate reads of the same fragments (identical read ids, row for row), so treat them as two views of one test, not independent replicates.
- Classes are unbalanced in genomes: the smoke sample had 45 nonpathogenic and 305 pathogenic training genomes, and only 6 and 39 in validation and test. Results rest on few nonpathogenic genomes, so read the genome-level intervals.
- Word weights (TF-IDF) and Word2Vec vectors are learned from `train` only; other splits are transformed with what training learned. Test results are produced only for the validation-chosen setting of each model.

## Run it

This stage needs no GPU and no new dependencies. On the machine you run on (for example over SSH):

```bash
git clone <repository-url> csf407_project
cd csf407_project
python3 -m venv .venv                      # Python 3.9–3.12
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest                 # optional check
.venv/bin/python run.py task4
```

`run.py task4` downloads the source files (about 2 GB, verified by MD5 and kept in `data/raw/`), samples the splits, learns the representations, runs every baseline and model, and writes the results. Disk use is about 2 GB for the downloads, plus a few GB of cached representations at the default sizes. Run it inside `tmux` or `nohup` so a dropped connection does not stop it:

```bash
nohup .venv/bin/python run.py task4 > task4.log 2>&1 &
```

Run the stages separately:

```bash
.venv/bin/python run.py prepare-splits     # download and sample; reuses a matching existing sample
.venv/bin/python run.py simple-models      # features, models, tables, figures, report
```

### Quick check first

A small run on the real data finishes quickly once the files are downloaded (the download dominates). The smoke test below used 4,000 training reads and took about 80 seconds for the models on an 8-core laptop:

```bash
.venv/bin/python run.py prepare-splits --splits-dir data/processed/task4_smoke \
    --train-per-class 2000 --val-per-class 1000 --test-per-class 1000
.venv/bin/python run.py simple-models --splits-dir data/processed/task4_smoke \
    --output-dir results/task4/smoke --feature-dir data/features/task4_smoke \
    --seeds 42 43 --trees 100 --epochs 5
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--train-per-class`, `--val-per-class`, `--test-per-class` | 50000, 10000, 10000 | Reads per class in each split (`prepare-splits`, `task4`). The test size applies to each of `test_1` and `test_2`. |
| `--splits-dir` | `data/processed/task4` | Where the split CSVs and manifest are saved. |
| `--force` | off | Replace existing splits that were made with different settings. |
| `--k` | `3 4 6` | Word sizes. |
| `--models` | all five | Any of `logistic_regression decision_tree random_forest naive_bayes gradient_boosting`. |
| `--seeds` | `42 43 44` | The first seed drives tuning and Word2Vec; later seeds repeat only the models that depend on a seed (trees, forests, boosting). |
| `--n-jobs` | all cores | CPU workers for forests and boosting. |
| `--workers` | 1 | Word2Vec workers. Values above 1 are faster but results are no longer exactly repeatable. |
| `--trees` | 200 | Trees per random forest. |
| `--bootstrap` | 100 | Genome-level resamples per interval; 0 disables intervals. |
| `--epochs`, `--vector-size`, `--window` | 10, 64, 5 | Word2Vec settings, as in steps 1–3. |
| `--output-dir` | `results/task4/simple_models` | Use a different directory to keep several runs. |
| `--feature-dir` | `data/features/task4` | Cache of learned representations; reused when the splits and settings match. |

The full-size run has not been timed. Word2Vec on 100,000 training reads and random forests on 4,096-column tables are the slow parts; use the quick check above to measure your machine, and reduce `--train-per-class` or `--seeds` if needed. A rerun with the same settings skips Word2Vec by reusing the cache.

## What is compared

- **Representations:** word counts, TF-IDF, average Word2Vec vectors, and TF-IDF-weighted average Word2Vec vectors, each at the chosen word sizes.
- **Models:** logistic regression, decision tree, random forest, multinomial naive Bayes (counts and TF-IDF only, since vectors can be negative), and histogram gradient boosting (only for tables of at most 1,024 columns, so not for 4,096-column six-letter counts).
- **Baselines:** majority class; G and C fraction only; letter composition with one-letter and two-letter words; number of unknown letters. These show how much of any result is available without word-level information. Read length is constant (250), so it cannot be a shortcut.
- **Shuffled-label control:** logistic regression trained on randomly reassigned training labels, repeated five times. Its average should sit near 0.5 AUC. A single shuffle can land away from 0.5 when one strong direction separates the classes, so read the average and spread; a consistently high value would indicate a leak.

AUC is the main metric. Accuracy, F1, and balanced accuracy use a 0.5 cutoff chosen without regard to differences between species.

## Where the results are

| File in `results/task4/simple_models/` | Contents |
| --- | --- |
| `report.md` | Data summary, baselines and control, models ranked by validation AUC, figures, reading notes |
| `results_summary.csv` | One row per configuration, averaged over seeds, with genome-level intervals |
| `results.csv` | One row per configuration and seed, including train, validation, and test metrics |
| `tuning.csv` | Every hyperparameter setting tried, **validation scores only** |
| `summary.json` | Settings, software versions, split manifest, runtime |
| `figures/model_heatmap.png` | AUC for every model, representation, and word size on validation, test 1, and test 2 |
| `figures/baseline_comparison.png` | Best model per representation (chosen on validation) against baselines and the control |
| `figures/generalisation.png` | Validation AUC against test AUC for every configuration |

Rank models using validation AUC; picking the best row by test AUC would tune on the test set. Intervals resample whole genomes within each class because reads from one genome are not independent; they remain rough with so few genomes.

Nothing in `data/` is committed (`data/raw/`, `data/processed/`, and `data/features/` are excluded). Decide separately whether to commit `results/`.

## Check the code

```bash
.venv/bin/python -m pytest
```

New checks cover split sampling and genome separation, agreement of the fast word counter with the steps 1–3 word maker, training-only learning of representations, metric calculations, every model, and a full synthetic run that writes all tables and figures.

# Task 4, part 2: neural models

Small neural networks trained on the same species-disjoint DeePaC splits and scored with the same metrics, genome-level intervals and row format as the simple models, so the two summaries can be compared directly. Torch is an optional extra: nothing above needs it.

| Model | Input | Notes |
| --- | --- | --- |
| `mlp` | weighted word counts (k = 3, 6) | two hidden layers |
| `cnn` | one-hot reads | DeePaC-style convolutions with a global max, no word size |
| `gru` | overlapping words (k = 3, 6) | bidirectional GRU, embeddings start from the train-only Word2Vec vectors |
| `transformer` | overlapping words (k = 3, 6) | 2 or 4 pre-norm layers, width 128, trained from scratch |

All are small (roughly 0.15–2 million values). The CNN, GRU and transformer see both strands: strands are flipped at random during training and the scores of a read and its reverse complement are averaged when predicting. One setting per model is compared on validation AUC, early stopping and the choice of epoch use validation reads only, the winner is retrained with the other seeds, and the test sets are scored once. A CNN trained on shuffled training labels is the leak control. Pretrained DNA models are not included yet.

## Run it on the GPU machine

Run the simple models first (see above), because the neural report compares against `results/task4/simple_models/results_summary.csv` when it exists and the two share the learned word vectors in `data/features/`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
# RTX PRO 6000 Blackwell needs a CUDA 12.8 build of torch 2.7 or newer:
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_capability(), (torch.ones(8, 8, device='cuda') @ torch.ones(8, 8, device='cuda')).sum().item())"   # expect (12, 0) 512.0
nohup python run.py task4 > task4.log 2>&1 &            # skip if already done
nohup python run.py neural-models > neural.log 2>&1 &
```

Quick check on any machine (about 10 minutes on a laptop; uses the small sample from the quick check above):

```bash
python run.py neural-models --splits-dir data/processed/task4_smoke --output-dir .cache/neural_check --max-epochs 3 --seeds 42 43 --bootstrap 20
```

Options: `--models` (any of mlp cnn gru transformer, to run one tier at a time), `--k`, `--seeds`, `--max-epochs` (15), `--patience` (3), `--batch-size` (256), `--device` (auto picks CUDA, then Apple MPS, then CPU), `--n-jobs`, `--simple-results`. The full run has not been timed.

Outputs go to `results/task4/neural_models/`: `report.md`, `results_summary.csv`, `results.csv`, `tuning.csv`, `history.csv` (every epoch), `summary.json` and `figures/` (`neural_comparison.png`, `training_curves.png`). GPU training is not bit-for-bit repeatable; seeds give spread, not exact reruns.

Tests that need torch are skipped when it is not installed: `python -m pytest -q`.
