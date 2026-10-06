# Results

Task 4 results are written here by `run.py`; steps 1–3 results stay in `outputs/`.

| Directory | Written by | Contents |
| --- | --- | --- |
| `task4/simple_models/` | `run.py simple-models` or `run.py task4` | `report.md`, `results_summary.csv`, `results.csv`, `tuning.csv`, `summary.json`, and `figures/` |

`tuning.csv` holds validation scores only. Test scores appear in `results.csv` and `results_summary.csv`, and only for the setting that validation selected.

Use `--output-dir` to keep several runs side by side, for example `results/task4/simple_models_small`.
