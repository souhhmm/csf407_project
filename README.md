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

Open **`outputs/report.md`** for the results, explanations, and plots. The eight plots use plain backgrounds, simple colours, and short labels. More detailed results are saved as tables rather than additional plots.

`requirements.txt` records the main packages and compatibility pins. `requirements-lock.txt` records every installed dependency's version. The chosen versions were checked on this Mac with Python 3.9.6.

After the first setup, rerun with `.venv/bin/python run.py all`. The first run downloads about 202 MB of source data. The local environment, downloaded data, and generated outputs are excluded from Git and can be recreated using these commands.

## Run the stages separately

Prepare the default dataset:

```bash
.venv/bin/python run.py prepare
```

Run analysis on the prepared data:

```bash
.venv/bin/python run.py analyze
```

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
| `outputs/report.md` | Plain-language results with eight minimal figures |
| `outputs/summary.json` | Settings, software versions, context comparisons, and runtime |
| `outputs/sequence_statistics.csv` | Lengths and letter proportions for each sequence |
| `outputs/class_statistics.csv` | Sequence summaries by class |
| `outputs/k_comparison.csv` | Comparison of word sizes |
| `outputs/k_3/`, `k_4/`, `k_6/` | Word tables, numeric representations, coordinates, and learned word-vector models |

Within each word-size directory, `row_ids.csv` gives the sequence order, and `vocabulary.json` gives the word order. `counts.npz` and `tfidf.npz` store word-based sequence representations without wasting space on zeros. `word_vectors.npy` stores the learned word vectors; `mean_vectors.npy` and `weighted_vectors.npy` store whole-sequence vectors. `idf.npy` stores the learned distinctive-word weights.

The two `context_pairs_*.csv` files contain the surroundings comparison. `nearest_words.csv` includes learned similarities and the fraction of matching letters. `common_words.csv` and `rare_words.csv` contain the highest and lowest observed counts. All observed words appear in `word_frequencies.csv`.

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
.venv/bin/python -m black --workers 1 run.py dna_language tests
```
