import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data import (
    DATASET_URL,
    curate_candidates,
    download_source,
    file_digest,
    load_dataset,
    read_fasta,
)


SPLIT_SOURCES = {
    "train": (
        {
            "filename": "nonpathogenic_train.fasta.gz",
            "label": "nonpathogenic",
            "md5": "78596e6a34ec266f1e86804f9e44f9c2",
        },
        {
            "filename": "pathogenic_train.fasta.gz",
            "label": "pathogenic",
            "md5": "760306b929c00e9072da6e1f0d82df20",
        },
    ),
    "val": (
        {
            "filename": "nonpathogenic_val.fasta.gz",
            "label": "nonpathogenic",
            "md5": "696379f88117ddd2f4512d078abde17f",
        },
        {
            "filename": "pathogenic_val.fasta.gz",
            "label": "pathogenic",
            "md5": "6b2e5c71634bd48a4c607e0979c09b2b",
        },
    ),
    "test_1": (
        {
            "filename": "nonpathogenic_test_1.fasta.gz",
            "label": "nonpathogenic",
            "md5": "e34306ddb3b693eff59448f6d8f7fab5",
        },
        {
            "filename": "pathogenic_test_1.fasta.gz",
            "label": "pathogenic",
            "md5": "b79795f8c27e01eeee04e0723f3c2a1c",
        },
    ),
    "test_2": (
        {
            "filename": "nonpathogenic_test_2.fasta.gz",
            "label": "nonpathogenic",
            "md5": "d8e7d711da63f17a60bb7e543cf8e53a",
        },
        {
            "filename": "pathogenic_test_2.fasta.gz",
            "label": "pathogenic",
            "md5": "f2af6eb49cffec82f8f426b6d7e478be",
        },
    ),
}
SPLIT_NAMES = tuple(SPLIT_SOURCES)
EVALUATION_SPLITS = ("val", "test_1", "test_2")
SPLIT_FIELDS = [
    "id",
    "sequence",
    "label",
    "genome",
    "source_file",
    "source_header",
    "source_split",
]
READ_LENGTH = 250


def genome_from_header(header: str) -> str:
    """Extract the source genome accession from a DeePaC read header.

    Args:
        header: FASTA header, such as ".../GCA_000003215.1.fq.000000007 ".

    Returns:
        The accession (GCA_000003215.1), or the whole last path part if the
        header does not follow the DeePaC naming scheme.
    """
    name = header.strip().rsplit("/", 1)[-1]
    return name.split(".fq.")[0] if ".fq." in name else name


def sample_split_records(
    path: Path, count: int, seed: int, label: str, split: str
) -> Tuple[List[dict], dict]:
    """Select a random candidate sample, reading every source record once.

    Unlike the validation sampler used in steps 1–3, this skips per-record
    cleaning so that the 10-million-read training files remain practical.

    Args:
        path: A local FASTA source.
        count: Maximum number of candidate records to retain.
        seed: Fixed random setting.
        label: Label supplied by the source file.
        split: Name of the published split the file belongs to.

    Returns:
        Candidate records and an audit of all records read.

    Raises:
        ValueError: If the file has no usable reads.
    """
    valid = re.compile("[ACGTRYSWKMBDHVN]{" + str(READ_LENGTH) + "}")
    rng = random.Random(seed)
    selected = []
    audit = Counter()
    for header, raw_sequence in read_fasta(path):
        audit["records_read"] += 1
        sequence = raw_sequence.upper()
        if not valid.fullmatch(sequence):
            audit["rejected_wrong_length_or_letters"] += 1
            continue
        audit["eligible_records"] += 1
        seen = audit["eligible_records"]
        position = seen - 1 if seen <= count else rng.randrange(seen)
        if position < count:
            record = {
                "id": label + "_" + split + "_" + str(audit["records_read"]),
                "sequence": sequence,
                "label": label,
                "genome": genome_from_header(header),
                "source_file": path.name,
                "source_header": header,
                "source_split": split,
            }
            if seen <= count:
                selected.append(record)
            else:
                selected[position] = record
        if audit["records_read"] % 1000000 == 0:
            print(
                "  read " + str(audit["records_read"]) + " records from " + path.name,
                flush=True,
            )
    if not selected:
        raise ValueError(
            "No usable " + str(READ_LENGTH) + "-letter reads in " + str(path)
        )
    return selected, dict(audit)


def audit_splits(records_by_split: Dict[str, List[dict]]) -> dict:
    """Check that the splits share no genomes and count repeated sequences.

    Args:
        records_by_split: Records for each split, each with sequence and genome.

    Returns:
        Genome counts per split, genome overlaps, and exact-sequence overlaps.
        The two test sets are mate reads of the same fragments, so their genome
        overlap is expected and not reported; their read pairing is counted.

    Raises:
        ValueError: If a training, validation, or test split shares a genome.
    """
    genomes = {
        name: {record["genome"] for record in records}
        for name, records in records_by_split.items()
    }
    sequences = {
        name: {record["sequence"] for record in records}
        for name, records in records_by_split.items()
    }
    names = list(records_by_split)
    genome_overlaps = {}
    sequence_overlaps = {}
    for first_number, first in enumerate(names):
        for second in names[first_number + 1 :]:
            pair = first + " & " + second
            sequence_overlaps[pair] = len(sequences[first] & sequences[second])
            if {first, second} == {"test_1", "test_2"}:
                continue
            genome_overlaps[pair] = len(genomes[first] & genomes[second])
    shared = {pair: count for pair, count in genome_overlaps.items() if count}
    if shared:
        raise ValueError(
            "Splits share genomes, so evaluation would not be on unseen species: "
            + json.dumps(shared)
        )
    paired = None
    if {"test_1", "test_2"} <= set(records_by_split):
        first = {record["source_header"] for record in records_by_split["test_1"]}
        second = {record["source_header"] for record in records_by_split["test_2"]}
        paired = {
            "test_1_reads": len(first),
            "test_2_reads": len(second),
            "reads_in_both": len(first & second),
        }
    return {
        "paired_test_reads": paired,
        "genomes_per_split": {name: len(found) for name, found in genomes.items()},
        "genome_overlaps": genome_overlaps,
        "exact_sequence_overlaps": sequence_overlaps,
    }


def prepare_splits(
    data_dir: Path,
    splits_dir: Path,
    sizes: Dict[str, int],
    seed: int = 42,
    force: bool = False,
    sources: Optional[dict] = None,
) -> Path:
    """Prepare balanced samples of the published DeePaC train, val, and test files.

    Args:
        data_dir: Parent directory whose raw subdirectory holds the downloads.
        splits_dir: Directory for the processed split CSVs and manifest.
        sizes: Sequences to keep per class for each split name.
        seed: Fixed random setting for candidate sampling and final selection.
        force: Whether to replace existing processed splits.
        sources: Published files for each split; defaults to SPLIT_SOURCES.

    Returns:
        The splits directory, reusing it if its settings and fingerprints match.

    Raises:
        ValueError: If settings are invalid, existing splits differ, or splits
            share genomes.
    """
    sources = SPLIT_SOURCES if sources is None else sources
    if set(sizes) != set(sources):
        raise ValueError("Give a size for each of: " + ", ".join(sources))
    if min(sizes.values()) < 1:
        raise ValueError("Sequences per class must be at least 1.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("The random setting must be between 0 and 4294967295.")
    manifest_path = splits_dir / "manifest.json"
    targets = {name: splits_dir / (name + ".csv") for name in sources}
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("per_class") == sizes
            and manifest.get("seed") == seed
            and all(
                path.exists() and manifest["csv_sha256"].get(name) == file_digest(path)
                for name, path in targets.items()
            )
        ):
            print("Using the existing checked splits: " + str(splits_dir), flush=True)
            return splits_dir
        raise ValueError(
            "The existing splits have different settings or content. Use --force to replace them."
        )
    if any(path.exists() for path in targets.values()) and not force:
        raise ValueError(
            "Split files exist without a manifest. Use --force to replace them."
        )
    records_by_split = {}
    source_audits = {}
    for name, files in sources.items():
        candidates = []
        source_audits[name] = []
        for number, source in enumerate(files):
            path = download_source(source, data_dir / "raw")
            records, audit = sample_split_records(
                path, sizes[name] * 2, seed + number, source["label"], name
            )
            candidates.extend(records)
            source_audits[name].append(
                {
                    **source,
                    "url": DATASET_URL + "/files/" + source["filename"] + "?download=1",
                    "audit": audit,
                }
            )
        records, audit = curate_candidates(candidates, sizes[name], seed)
        records_by_split[name] = records
        source_audits[name].append({"sample_audit": audit})
        print(
            "Selected "
            + str(len(records))
            + " reads for "
            + name
            + " from "
            + str(len({record["genome"] for record in records}))
            + " genomes",
            flush=True,
        )
    split_audit = audit_splits(records_by_split)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, records in records_by_split.items():
        temporary = targets[name].with_suffix(".csv.part")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPLIT_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(targets[name])
    manifest = {
        "dataset": "DeePaC IMG bacterial DNA reads, published train, validation, and test files",
        "dataset_url": DATASET_URL,
        "paper_url": "https://doi.org/10.1093/bioinformatics/btz541",
        "per_class": sizes,
        "seed": seed,
        "sampling": "random reservoir candidates (2x requested) per file, deduplication, balanced random selection",
        "sources": source_audits,
        "split_audit": split_audit,
        "csv_sha256": {name: file_digest(path) for name, path in targets.items()},
        "label_meaning": "Labels belong to source organisms, not to individual disease-causing fragments.",
        "test_sets": "test_1 and test_2 are the two mate reads of the same fragments (the same read ids), so they are two views of one test, not independent replicates.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared splits: " + str(splits_dir), flush=True)
    return splits_dir


def load_splits(splits_dir: Path) -> Tuple[Dict[str, List[dict]], dict]:
    """Load prepared split CSVs and their manifest.

    Args:
        splits_dir: Directory written by prepare_splits.

    Returns:
        Checked records for each split and the manifest.

    Raises:
        ValueError: If the manifest or a split file is missing or a file's
            fingerprint no longer matches the manifest.
    """
    manifest_path = splits_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            "No prepared splits in "
            + str(splits_dir)
            + ". Run run.py prepare-splits first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_by_split = {}
    for name, digest in manifest["csv_sha256"].items():
        path = splits_dir / (name + ".csv")
        if not path.exists() or file_digest(path) != digest:
            raise ValueError(
                "Split file " + str(path) + " is missing or changed since preparation."
            )
        records_by_split[name], _ = load_dataset(path)
    return records_by_split, manifest
