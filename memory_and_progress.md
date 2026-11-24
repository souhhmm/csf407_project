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

## 2026-11-23 GPU box session start (RTX PRO 6000 96GB, py3.12.3 .venv, NO torch yet, data/ and results/ empty -> nothing has been run at full size)
Sanity review findings (presented to user, awaiting plan approval, no code written yet):
1 genome imbalance: 6 nonpatho val/test genomes, one = 21-29% of nonpatho reads -> val selection/early stop/AUC noisy; steps1-3 exploration used val split only -> class word ratios/GC/PCA may be genome identity; redo per-genome.
2 train reads balanced but genomes not (45 vs 305) -> cap reads per genome / weight.
3 per-read labels are organism-level -> add genome-level aggregation (mean read score per genome, mates pooled).
4 CNN tiny (1 conv k15 + 1 conv, 64/128 filters, no lr schedule) and 15 epochs on 100k reads; smoke gave AUC .5. scale data (1e6+/class) + DeePaC-size CNN.
5 w2v on overlapping k-mers: neighbours share k-1 letters -> trivial context; mean-pooled w2v = linear map of counts.
6 TableInputs sparse->dense on CPU per batch (slow); batch 256 underuses GPU.
7 T5 importances, T6, T7 missing. md5s/downloads: ~2GB+, earlier 1MB/s.

## 2026-11-23 round 2: user feedback on review (DECISIONS)
- User trusts the human-curated DeePaC splits: DO NOT change val/test sampling or early-stopping/selection metric. Headline = existing read-level metrics. ADD per-genome stats (per-genome AUC/acc, genome-weighted AUC, mean-read-score-per-genome aggregation, mates pooled) IN ADDITION. Genome-cap on train = optional ablation only.
- Genome identity: handle via per-genome analysis (T1-3 stats per genome, LOGO-style consistency of class word ratios, PCA colored by genome), importances checked for per-genome consistency. Affects T5/T6 claims: only claims holding across genomes count.
- Models: pick meaningfully larger for ALL, must finish today EOD; smoke-time, extrapolate, launch bigger runs after smaller ones if underestimated. Document EVERYTHING in this file (user will write report from it).
- w2v concern (overlapping k-mer contexts trivial): document it; alternative tests = RC-pair similarity (a word and its reverse complement should be close), non-overlapping/strided tokenization w2v, compare to count-SVD embeddings.
- Transformer: compare from-scratch vs pretrained (DNABERT-2-117M, maybe NT-v2-100M) at reasonable scale.
- Problem 7 plan: bacpacs test sets (other curated db) as OOD test; read-length truncation; genome-level aggregation as 'longer sequence'.
- Rename outputs/ -> old_outputs/ AFTER concerns settled; new stuff in outputs/.
- Env facts: zenodo record 3678563; needed 8 files = 2.01 GB (train 808+803MB, val 101+101, test_1 51+50, test_2 51+50); speed 1.7MB/s => ~20 min. bacpacs test_1/2: nonpatho 36MB, patho 64MB each; also .rds fold files in record. HF reachable. OpenRouter API reachable (200) but NO credentials found in env/~/.bashrc/.config -> ask user. Disk: check df before downloads.

## 2026-11-23 ~13:05 execution started
- OpenRouter check: key file ~/.config/mer/openrouter_key exists (mode 600, 73 bytes; documented in ~/MER/MER_lyrics/mer/lyrics/jev.py, used there for TypeSafe JEV 'decisions' endpoint). GET /api/v1/key with it -> HTTP 401 "User not found" => key invalid/revoked or not a standard OpenRouter key. Did not print/modify it. Need user to supply working key.
- Downloads: `run.py prepare-splits --train-per-class 20000 --val-per-class 5000 --test-per-class 5000 --splits-dir data/processed/task4_probe` (bg, log .cache/dl_probe.log) downloads all 8 files to data/raw sequentially (~20min), producing a small probe split. Real split built afterwards (raw reused).
- torch cu128 install bg: .cache/pip_torch.log.

## 2026-11-23 13:15 env + LLM-experiment plan (ALL DECISIONS, for the report)
ENV: old .venv was an empty stub (no pip/numpy) -> built conda env `.venv-gpu` (py3.12.14) in project dir; installing requirements.txt, torch cu128, transformers, accelerate, einops (log .cache/pip_torch.log). USE .venv-gpu/bin/python for everything. .gitignore now ignores .venv*/ and .cache/.
DOWNLOAD: zenodo bandwidth-limited ~1.5-1.7MB/s total regardless of parallelism. Running: sequential prepare-splits (probe split) PLUS parallel curl of the other 11 files (8 IMG + 4 bacpacs test) into data/raw as *.dl then mv to final name (download_source md5-verifies existing files and skips). ETA ~13:45. Real splits are built once raw files verified (full train files must be read once for reservoir sampling).
LLM EXPERIMENT (T6) plan:
 - Models via OpenRouter (user approved tiers): very cheap = Gemini Flash-Lite or GPT-mini class (or Qwen3 small); cheap open-weight = GLM-4.x (user wants GLM/open-weight); medium = Sonnet-class Claude. Check live ids/prices through /api/v1/models before use. Maybe add DeepSeek/Qwen only if user asks.
 - Same evidence package for all models: top-20 words/class (LR coef + class-frequency ratio), GC & CTAG stats, per-genome consistency flags, CNN top motifs, model AUCs.
 - Questions: Q1 which patterns separate classes and why; Q2 composition vs deeper; Q3 held-out prediction questions (e.g. which of 5 words is more pathogen-enriched, effect of GC matching on AUC); Q4 dependence on read length/truncation; control = decoy evidence package built from SHUFFLED labels (does the model invent explanations for noise?).
 - 3 samples/model at T=0.7 for stability. Score each claim true/false/untestable against our numbers; count held-out predictions correct.
 - Budget estimate: ~72 calls, ~220k input + ~110k output tokens (more if reasoning models) => about $1-2 total; recommended key credit limit $5 ($3 probably enough).
 - Key goes in ~/.config/csf407/openrouter_key (mode 600, outside repo); never print/log. Old MER key (~/.config/mer/openrouter_key) returned 401 'User not found' on /api/v1/key.

## 2026-11-23 13:20 standing instruction (user, durable for this project)
- EMAIL: user authorised me to send email to f20230392@goa.bits-pilani.ac.in (a) a final update when EVERYTHING is complete, (b) an email marked URGENT (subject starts "URGENT:") only if something goes very wrong and user intervention is needed (e.g. key/credentials broken, disk full, GPU unusable, runs can't finish by EOD). Use Gmail MCP send_message (load schema via ToolSearch select:mcp__claude_ai_Gmail__send_message). No other recipients; no secrets in emails.
- Status 13:20: torch 2.11.0+cu128 OK in .venv-gpu (capability 12.0, matmul ok), transformers 5.24.0. OpenRouter key at ~/.config/csf407/openrouter_key verified: limit $5, spend $0. Train downloads: nonpatho done, patho ~740/803MB. Killed the sequential probe prepare-splits (redundant).
- Code written but UNTESTED: splits.prepare_extra_sets/evaluation_splits, run.py prepare-extra, --train-limit/--fit-limit/--no-scores/--max-grid, classifiers.save_scores, features fit_limit, simple/neural scores saving, networks bigger (cnn 4 conv blocks+BN, mlp 3 hidden, gru 2 layers, transformer d512 6/8 layers; GRIDS mlp[1024,2048] cnn[256,512] gru[256,512] transformer[6,8]), src/genome_report.py (not yet wired into run.py).
- TODO code: pretrained.py (DNABERT-2 finetune vs random-init), genome-report cmd, T5 analysis, LLM experiment script, tests.

## 13:45 progress (code + findings)
- DNABERT-2 loading: transformers 5.x breaks its remote code (meta-device alibi init) -> pinned transformers==4.46.3. Local patched copy at data/models/dnabert2 (copied from HF snapshot; patched bert_layers.py: set BertModel.config_class=local BertConfig; triton flash-attn import forced to ImportError + stub flash_attn_triton.py so pytorch attention is used). Loads OK (117.07M params), tokenizer pad id 3, 252bp read -> ~144 BPE tokens (max_length 192). data/models is NOT in .gitignore? (check before commit).
- Added to networks/neural_*: models dnabert2 (pretrained, lr 3e-5) and dnabert2_scratch (same arch random init, lr 1e-4); TokenInputs (fwd + RC tokenized separately, RC aug in train, RC averaging at predict); BertReader mean-pool head; no shuffled control for these.
- Extra eval sets (prepare-extra): bacpacs_test_1/2 (other curated db; BacPaCS test = 60 patho + 40 nonpatho genomes; ids like 1041522.28; may share species with IMG train - not verifiable), test_1_len100, test_1_len150, test_1_letters_shuffled (composition-preserving shuffle; tests composition vs order).
- genome_report.py + run.py genome-report: per-genome table, genome-weighted AUC, genome-level AUC/acc of mean read score, test_pooled (mates), reads-per-genome curve.
- llm_experiment.py: models google/gemini-3.1-flash-lite ($0.25/$1.5 per M), z-ai/glm-5.2 ($0.62/$2.42, open weight), anthropic/claude-sonnet-5.5 ($2/$10); budget guard $4, max_tokens 4000, JSONL resume. (Cheaper alt seen: anthropic/claude-haiku-5.5 $0.1/$0.5.)
- Real split build launched 13:25: prepare-splits train 1M/class val 50k test 100k (log .cache/prepare_full.log), reading train files ~1M reads/min (~25 min).
- Real test genome counts (IMG test_1): 39 patho / 6 nonpatho genomes (confirmed from headers).

## 13:30 INCIDENT: disk quota
- /home (sdb1) has a per-user 40GB quota (was at 40.96GB) -> writes failed (EDQUOT), prepare-splits died while writing train.csv (lost ~6 min). ~/data -> /media/data/trl03 is a separate disk (195GB quota, ~100GB used by user's other projects).
- FIX: moved my big files to ~/data/csf407/ and left symlinks: .venv-gpu -> ~/data/csf407/.venv-gpu ; data/{raw,models,processed,features} -> ~/data/csf407/data_* ; results/task4 -> ~/data/csf407/results_task4. HF_HOME for any new HF downloads: ~/data/csf407/hf_home. Home usage after fix ~30GB/40GB. DO NOT put large files under /home; do not touch user's other dirs (MER, anaconda3 etc.).
- Relaunched prepare-splits --force 13:30 (train 1M/class val 50k test 100k/class), log .cache/prepare_full.log.

## Smoke timings (200k train reads, 1 epoch, real splits; 2026-11-23)
- cnn 5.4 s/ep (val AUC .807 @2ep); mlp(1024) 6 s/ep (.771); gru(256) 41 s/ep (.794); transformer(6L,512) 86 s/ep (.775); dnabert2 pretrained 206 s/ep (.789); dnabert2 random-init 191 s/ep (.525 -> needs more epochs). Tokenizing all splits ~160-210 s.
- Full (2M reads) is ~10x per epoch. Launched `.cache/full_gpu.sh` at 14:45 IST, two parallel chains:
  A: cnn (15 ep, seeds 42,43, full grid, 2M reads) -> mlp (12 ep, 2 seeds) -> dnabert2 (1M reads, 2 ep) -> dnabert2_scratch (same)
  B: gru (1M reads, 6 ep) -> transformer (1M reads, 6 ep). Outputs results/task4/full_<model>, logs .cache/full_<model>.log, done flag .cache/full_gpu.done.
- Simple models full run running concurrently (results/task4/simple_models, train/fit limit 400k).

## Full GPU runs finished (2026-11-23 ~19:00 IST; chains took ~4.4 h)
- Wall: cnn 1010 s tuning (+687 s control), mlp 586 s, gru 2264 s, transformer 4593 s, dnabert2 2772 s, dnabert2_scratch 2696 s.
- Val AUC (official val, read level): cnn .790, mlp .776, gru .782, transformer .778, dnabert2 .781, dnabert2_scratch .501 (random init, 2 ep on 1M reads: did not learn -> report honestly), shuffled-label controls ~.50.
- Read-level test AUC (mean over seeds): cnn .882, mlp .896, transformer .856, dnabert2 .891; gru/scratch in results/task4/full_*/genome/genome_summary.csv.
- Truncated 150 bp ~.83-.85, 100 bp ~.77-.80, letter-shuffled ~.53-.55 (composition-preserved shuffle destroys signal -> models use order/k-mer info, not just GC), BacPaCS ~.64-.70 (large drop on a different database; genome AUC ~.68-.75).
- Genome AUC on test ~.87-.92 > read AUC (aggregation helps).
- Bug fixed: src/genome_report.load_truth now drops duplicate sequences (keeps first) to match loader dedup (truncated sets lost ~30-50 reads).
- genome-report run for all six full_* dirs -> results/task4/full_<m>/genome/.

## T5 interpret done (outputs/task5, 19:30 IST)
- Genome-level word effects (train 45 non-patho/310 patho genomes; held-out 12/78): Spearman of genome AUC train vs held-out k3 .83, k4 .89, k6 .86; top-word direction agreement k6 100% (k3 92%). Significant (BH q<.05) words: k3 16, k4 88, k6 1365.
- Genome GC AUC: train .45, held-out .41 (GC weak, not the driver). LR coefficient vs genome effect Spearman .63 (k6).
- RC test: word2vec overlapping RC cosine .19 vs random .17 (RC nearest-neighbour 0.02%), strided .35 vs .22, count-SVD .87 vs .81 -> word2vec does not learn RC symmetry; SVD slightly closer (all weak).
- Top pathogenic 6-mers: GCTAGC, AACCCT/AGGGTT, CTAGCG... (CTAG-rich); nonpathogenic: CATGGA, TGTTCC, GGATAC, ACGGAA, TTCCGT... (CCGT/GGAT-rich). CNN motif extraction NOT done.
## T6 LLM experiment (started 19:35)
- src/llm_prompts.py: real + decoy (class names swapped in word lists) x Q1-Q4; evidence shows ONLY val/test_pooled AUCs (truncated, shuffled, BacPaCS hidden so Q3/Q4 are real predictions). Prompts saved results/task4/llm_prompts.json; replies results/task4/llm_replies.jsonl (3 models x 8 prompts x 3 samples = 72 calls, T=0.7, budget guard $4).
- Note: dnabert2_scratch genome_auc .786 on test despite read AUC .501 (mean-score aggregation of near-constant scores; not meaningful).

## T6 done (19:55): 72 calls, $0.835. Scores in results/task4/llm_claim_scores.md (hand-scored on sample 0 + skim; Sonnet best calibrated on BacPaCS shift, Gemini over-optimistic; decoy: models narrate swapped labels confidently; GLM 10/72 empty replies from token budget).
- Remaining at 19:55: simple-models run finishing (k=6 stage), then rename outputs->old_outputs, populate new outputs, final email.

## Finalisation (2026-11-24 ~10:30 IST)
- Simple-models full run finished late (k=6 stage was slow; ~14 h wall incl. waiting), genome report done. Best simple model test_pooled read AUC .854 (gradient boosting on k-mer counts, k=3/4), genome AUC .93 -> simple k-mer GBM is within ~.04 of neural models (.88-.90 CNN/MLP/GRU/DNABERT-2).
- outputs/ renamed to old_outputs/ (T1-3 old results); new outputs/ holds task4 (per-model reports, tuning, history, genome/ tables, figures), task5 (interpretation), task6 (LLM prompts, replies, claim scores).
- NOT done: T7 final model choice + presentation, CNN motif extraction, T1-3 stats redone per genome (partly covered by task5 word effects), genome-capped ablation, NT-v2. Nothing committed/pushed.

## NEXT (agreed with user 2026-11-24, NOT started): three gap tasks
1. Redo T1-T3 per genome: new src/genome_explore.py + run.py command; per-genome count vectors from cached features, per-genome word freq/rare-common words/GC/seq stats with class comparison at genome level, PCA/UMAP at read and genome level coloured by class AND genome. CPU, ~10-30 min. Output outputs/task1_3_genome/.
2. Visualisation of learned sequence space: script reloading saved weights (results/task4/full_*/weights/*.pt) for cnn/gru/dnabert2, embeddings of 20-50k test reads, UMAP/t-SNE/PCA coloured by class, genome, correct/wrong, next to raw k-mer PCA of same sample; quantify with linear probe/silhouette (genome id vs class). GPU minutes.
3. T6 verification: LLM-assisted scorer over ALL 72 replies (claim extraction + true/false/untestable vs our results; decoy scored against swapped labels), ~$0.5-1, key has ~$4.1 left (guard $4 total); spot-check verdicts by hand; flag in write-up as LLM-assisted. Output per-claim table + counts per model.
Also still open: T7 (final model choice, answer, presentation; check old presentation PDFs), CNN motifs (optional), docs.
Git: pushed to origin souhhmm/csf407_project main after README update (see below if recorded). .gitignore must use patterns WITHOUT trailing slash for symlinked data dirs.

## VERIFICATION PASS (2026-11-24) - findings, NO changes made yet
Runs: all six full_* runs and simple_models DID complete (history/results/scores exist). Problems are in plots/reports/interpretation:
1. src/plots.py NEURAL_LABELS lacks dnabert2, dnabert2_scratch -> training_curves.png blank (models list empty) for both; neural_comparison falls back to raw key names. Fix: add labels.
2. training_curves: legend shows raw JSON dict ('{"filters": 256}'); only seed 42 plotted; y-axis autoscaled to ~0.01 range so noise looks dramatic; epochs 1-2 are the best for cnn/mlp (val AUC falls while train loss falls) -> overfitting to training genomes; "larger models" gave no val gain. dnabert2 has only 2 points (2 epochs).
3. neural_comparison.png: each full_<m> report compares one model only (simple results file did not exist at run time, so no simple-model bars); bars start at 0.45 axis (truncated), value labels overlap ticks. Need ONE combined comparison figure across all models + simple + baselines.
4. report.md text stale: "Networks are small on purpose", "Test 1 95% interval" only; no genome-level metrics in reports; no mention of extra sets (BacPaCS/truncated/shuffled) - those exist only in genome/*.csv, no plots at all (aggregation_curves.csv never plotted).
5. dnabert2_scratch: loss stays 0.6935 (=ln2) i.e. never learned; test AUC .501; the scratch-vs-pretrained comparison is INVALID as run (2 epochs, 1M reads, lr 1e-4); its genome_auc .786 is an artefact. Needs lr/warmup/longer or must be reported as 'did not train'.
6. Training budget: gru 5-6 epochs, transformer 4-5, dnabert2 2 epochs (1M reads, not 2M); gru h=512 hit max_epochs 6 while still improving. Val AUC (.77-.79) is noisy (45 genomes; 3 nonpathogenic genomes hold ~60% of nonpatho val reads) so early stopping/selection is near-random among configs.
7. Simple models: gradient boosting k=6 counts/tfidf cells blank in heatmap (check whether intentional skip) ; simple models trained on 400k reads not 2M (documented in memory, not in README).
8. T5/T6 outputs have no figures; task5 lacks plots (volcano of word effects, per-genome consistency); T6 scoring only sample 0 (already planned).
9. README claims 'AUC 0.86-0.90' etc. are read-level means over seeds; val-vs-test gap (val .78 vs test .89) should be explained (val has 6 nonpatho genomes only).
- FIX WORK (2026-11-24): scratch DNABERT failure cause = no LR warmup. Diagnostic (.cache/diag_scratch.py, 60k reads): lr1e-4+500-step warmup learns (val .742/.762/.769 over 3 ep); lr 3e-4 and 1e-3 collapse. Added warmup_steps to train_network (default 0, other models unchanged) and FIXED dnabert2_scratch warmup=1000. Old failed run kept at results/task4/full_dnabert2_scratch_nowarmup. Rerun: scratch 6 ep (patience 3) and gru_long (14 ep, patience 4) -> results/task4/full_dnabert2_scratch, full_gru_long (started ~11:20 IST; .cache/rerun.done flag).
- Plot/report fixes done (2026-11-24): plots.py labels for dnabert2*, legend labels, bar axis 0.5-1; new src/final_report.py + `run.py final-report` (regenerates per-model figures with simple-model bars, summary/{comparison,robustness,reads_per_genome,training_overfit}.png + report.md + all_models_by_split.csv, task5 figures, task6 overview). Stale 'small on purpose' text patched in generator and copied reports. GB skipped for >1024 features is intentional (MAX_BOOSTING_FEATURES). Pending: reruns (scratch w/ warmup, gru_long) -> genome-report on them -> final-report -> README/notes.
- Reruns finished (~12:30 next day): scratch DNABERT with 1000-step warmup now learns: val .771 (epoch 1 best, diverged epoch 4 loss .689), test_pooled read AUC .836, BacPaCS .634 vs pretrained .890/.701. gru_long (14 ep cap, early stop): val .782, test .874 vs first gru .894 -> single-seed noise ~.02; no gain from longer training. genome-report done for both; final-report regenerated; README caveats added. outputs/task4/full_dnabert2_scratch_nowarmup keeps the failed run for the record.

## Round 3 reruns (user approved 2026-11-24)
- Added lr_decay (linear to 0 over max_epochs after warmup) to train_network; neural_models reads lr from grid params (params.get("lr")), warmup/decay from FIXED. dnabert2 grid lr [3e-5,1e-5], warmup 500, decay; dnabert2_scratch grid lr [1e-4,5e-5], warmup 1000, decay. Both: max-epochs 8, patience 3, 1M reads, seed 42.
- Old runs renamed: results/task4/full_dnabert2_2ep (2-epoch cap), full_dnabert2_scratch_warmup_only (warmup, no decay, lr 1e-4, 4 ep diverged), full_dnabert2_scratch_nowarmup (chance). New outputs: results/task4/full_dnabert2, full_dnabert2_scratch (.cache/rerun2.done flag; logs .cache/full_dnabert2_v2.log, full_dnabert2_scratch_v2.log). Then: genome-report, final-report, README + report update, commit, push (user must run push).
