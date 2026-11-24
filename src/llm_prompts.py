import json
from pathlib import Path

import pandas as pd

QUESTIONS = [
    ("q1", "Which sequence patterns separate pathogenic from nonpathogenic bacteria here?"),
    ("q2", "Do the models only use composition (GC content, letter counts) or something deeper? Argue from the evidence."),
    ("q3", "Predict, with numbers, the AUC of the same models on a different curated database of unseen genomes, and justify."),
    ("q4", "How should accuracy depend on read length (100, 150, 250 bp) and on averaging many reads of one genome? Give expected AUCs."),
]


def top_words(path: Path, count: int = 20, decoy: bool = False) -> dict:
    """Top words per class, replicated on held-out genomes (held-out q < 0.1).

    Args:
        path: word_effects_k6.csv.
        count: Words per class.
        decoy: Swap the class names (the shuffled-label control).

    Returns:
        Words more frequent in each class.
    """
    table = pd.read_csv(path)
    table = table[table["q_value_held_out"] < 0.1]
    patho = table[table["genome_auc"] > 0.5].nsmallest(count, "q_value")["word"].tolist()
    non = table[table["genome_auc"] < 0.5].nsmallest(count, "q_value")["word"].tolist()
    if decoy:
        patho, non = non, patho
    return {"more_in_pathogenic": patho, "more_in_nonpathogenic": non}


def evidence(results_root: Path, task5: Path, decoy: bool = False) -> str:
    """Build the evidence text shown identically to every model.

    Args:
        results_root: Folder with full_<model> result directories.
        task5: Folder with interpretation outputs.
        decoy: Use swapped class labels in the word lists (control).

    Returns:
        The evidence as text.
    """
    summary = json.loads((task5 / "interpretation_summary.json").read_text())
    rows = []
    for model in ("cnn", "mlp", "gru", "transformer", "dnabert2", "dnabert2_scratch"):
        frame = pd.read_csv(results_root / ("full_" + model) / "genome" / "genome_summary.csv")
        frame = frame[~frame["tag"].str.contains("control|shuffled")]
        frame = frame[frame["split"].isin(("val", "test_pooled"))]
        frame = frame.groupby("split")[["read_auc", "genome_auc"]].mean().round(3)
        rows.append(model + ": " + json.dumps(frame.to_dict("index")))
    gc = summary["genome_gc_auc"]
    return (
        "Task: classify 250 bp bacterial DNA reads as from pathogenic or nonpathogenic genomes (labels at organism level).\n"
        "Train/val/test are genome-disjoint. 6-mer words more frequent per genome in each class (replicated on held-out genomes):\n"
        + json.dumps(top_words(task5 / "word_effects_k6.csv", decoy=decoy))
        + "\nGC content separates the classes weakly: genome-level AUC of GC = "
        + format(gc["train"], ".2f") + " (train), " + format(gc["held_out"], ".2f") + " (held out).\n"
        "Word-effect direction agreement train->held-out genomes: 100% for the top 6-mers.\n"
        "Reverse-complement test: embeddings do not treat a word and its reverse complement as similar (word2vec cosine 0.19 vs 0.17 random).\n"
        "Model AUCs (read-level, and genome-level via mean read score) on validation and test only. Other evaluation sets (truncated reads, letter-shuffled reads, a different curated database) are hidden from you on purpose; Q3 and Q4 ask you to predict them:\n"
        + "\n".join(rows)
    )


def prompts(results_root: Path, task5: Path) -> list:
    """Real and decoy prompts for every question.

    Args:
        results_root: Folder with full_<model> result directories.
        task5: Folder with interpretation outputs.

    Returns:
        Dicts with id and text.
    """
    out = []
    for decoy in (False, True):
        text = evidence(results_root, task5, decoy)
        for qid, question in QUESTIONS:
            out.append(
                {
                    "id": ("decoy_" if decoy else "") + qid,
                    "text": text + "\n\nQuestion: " + question
                    + " Make specific, checkable claims; say what would falsify each. Be concise (under 350 words).",
                }
            )
    return out
