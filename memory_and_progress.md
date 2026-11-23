# scratch (agent only)
rules: no push to origin (souhhmm/csf407_project); match repo style (docstrings Args/Returns/Raises, type hints, terse comments); no code until user OKs task-4 plan.
repo: csf407_project/ ; src/{data,sequences,representations,exploration,analysis,plots}.py, run.py cmds prepare|analyze|explore|all; 40 tests; py3.9 lock; numpy/scipy/pandas/sklearn/gensim/mpl only (no torch).
git: 4 commits, last "add further exploratory results". data/raw,processed gitignored -> NOT present locally; no venv; can't rerun (needs 202MB dl). verification = artifacts in outputs/ + README + deck.
data: DeePaC IMG val files, 5000/class =10k, 250bp, 2594 seqs w/ N; k=3,4,6 all vocab covered; w2v sg dim64 win5 ep10 neg5 seed42.
T1-3 verdict: done (BoW, TF-IDF, w2v, mean/tfidf-weighted pooled, PCA/SVD plots, freq tables, context check, GC corr ~0.996, class word ratios CTAG 5.3x, shuffle check). Caveats: w2v trained on all 10k (unsupervised, but incl. future test rows); sample from val split only; labels = organism pathogenicity; GC dominates coord1.
T4 key concerns: species-level leakage (random reads same genome) -> use DeePaC train/val/test species-disjoint files; GC/len confounds (len const=250 so len moot); w2v fit on train only; baselines: GC-only, shuffled-label, k-mer LR/DT/RF/NB; NN: CNN (DeePaC-like, RC-aug), BiLSTM; transformer from scratch vs pretrained (DNABERT-2/NT). Compute: Colab ok for k-mer+small CNN on ~1e5-1e6 reads; own GPU for large train + pretrained FT.
status: discussion phase with user.

## T4 build (user approved: use official DeePaC train/val/test; order simple->smoke->neural(later)->large; no push; README append; single script; results/ folder; torch ok; GPU RTX PRO 6000 96GB but keep models modest)
env: .venv built w/ ~/.pyenv/versions/3.12.0 from lock; 40 old tests pass (47s).
zenodo md5: nonpatho_train 78596e6a34ec266f1e86804f9e44f9c2 | patho_train 760306b929c00e9072da6e1f0d82df20 | nonpatho_test_1 e34306ddb3b693eff59448f6d8f7fab5 | patho_test_1 b79795f8c27e01eeee04e0723f3c2a1c | nonpatho_test_2 d8e7d711da63f17a60bb7e543cf8e53a | patho_test_2 f2af6eb49cffec82f8f426b6d7e478be | val md5 in data.py. (use IMG pathogenic_*/nonpathogenic_*, NOT bacpacs/_imb). train ~800MB each; total dl ~2GB.
headers: ">/home/.../trainingReads/pathogenic/GCA_000003215.1.fq.000000000 " -> genome acc parseable; files are ordered genome by genome (prefix = few genomes => reservoir over full file). test_1/test_2 = mates of same genomes (same read idx) -> report separately, never pool as independent.
advisor plan: new modules (splits.py, features.py fit-on-train-only, classifiers.py, simple_models.py), don't alter old prepare/sequences.csv; results in top-level results/ (analyze may wipe outputs/); val-only tuning, test touched only for val-chosen cfg; baselines majority/GC/k1/k2/ambig + shuffled-label ctrl; models LR/DT/RF/NB(count,tfidf only)/HGB(<=1024 feats); n_jobs exposed; genome-level bootstrap CI; torch separate later; tests tiers.
design: run.py cmds prepare-splits | simple-models | task4. data/processed/task4/{train,val,test_1,test_2}.csv + manifest. counts via numpy rolling base-4 (A0 C1 G2 T3) not python word lists.

## status (10:36 Oct 6)
DONE code: src/{splits,features,classifiers,simple_models}.py, plots.py (+3 fns), run.py (+prepare-splits,simple-models,task4), tests (test_splits/features/classifiers/simple_models + conftest fixture synthetic_sources), .gitignore +/data/features/, results/README.md. ~70 tests pass.
findings: shuffled-label ctrl is NOT ~0.5 on tiny/strong-signal data (random direction -> AUC near 0/1); now 5 permutations avg, report text explains. CI = genome bootstrap, bounds averaged over seeds.
RUNNING: bg `run.py prepare-splits --splits-dir data/processed/task4_smoke --train-per-class 2000 --val-per-class 1000 --test-per-class 1000` log .cache/smoke_prepare.log (download ~1MB/s ~35min, 2GB into data/raw). NEXT: smoke simple-models on it (--output-dir results/task4_smoke? keep out of results/task4), then README append (task4 section, SSH instructions, runtimes), black, full pytest, final advisor call, tell user (NOT pushed; nothing committed).

## 11:0x Oct 6: smoke on real data OK (80s models w/ 4k train reads; val AUC models ~.75-.79, k2 comp baseline .65, GC-only ~.44-.52, ctrl ~.47-.50). README task4 section appended. 70 tests pass, black+pyflakes clean. Nothing committed/pushed. KEY CAVEAT: genomes imbalance nonpatho 45 train/6 val+test genomes vs patho 305/39 (one nonpatho genome = 21-29% of nonpatho val/test reads). Full-size run untimed. Smoke artifacts in .cache/ + data/processed/task4_smoke (gitignored). TODO next: neural models (torch, separate req file), larger runs; maybe genome-stratified sampling option.

## Stage 2 (neural) status
- Pushed simple-models commit 829bc21 (user-approved, author = user, no Claude trailer). Do NOT commit/push neural stage without asking.
- Written, uncommitted: src/networks.py (mlp/cnn/gru/transformer, rc, kmer_tokens), src/neural_training.py, src/neural_models.py, plots (plot_neural_comparison, plot_training_curves), run.py `neural-models`, requirements-neural.txt, tests/test_neural.py (importorskip torch), features.load_or_build_representations (simple_models refactored to use it). README part 2 appended. 79 tests pass.
- Smoke (4k reads, 4 epochs, MPS, ~8 min; transformer slowest): mlp .795 val, gru k6 .789, transformer k3 .78, cnn ~.5 (undertrained, expected at 64 steps/epoch). Control cnn .52. Simple LR best val .793/test .825.
- Findings: mate test sets, genome imbalance (see earlier). Venv py3.12, torch 2.14.1.
- Todo: pretrained DNA models (tier 3, optional); full runs on SSH box; maybe CNN needs more epochs/lr check at full size; mate-averaging, genome-capped sampling.

## Remaining tasks (as of 2026-11-23)
- T1-3: done. T4: simple stage pushed (829bc21); neural stage committed+pushed 2026-11-23 (user-approved).
- T4 leftovers: full-size runs on GPU box; CNN epochs/lr check at full size; mate-averaging (test_1/test_2); genome-capped sampling (nonpatho genome imbalance); optional pretrained DNA models (DNABERT-2/NT).
- T5 (partial): GC/shuffle/unseen-genome checks exist. Missing: important-words/features per class (LR coefs, RF importances), "patterns vs composition" write-up.
- T6 (not started): AI-generated explanations of patterns, compare across AI tools, verify against computational results.
- T7 (not started): final model + answer to "can AI read DNA?" + final presentation (check whether current presentation.pdf / UhmmAkshually.pdf reflect T4).
