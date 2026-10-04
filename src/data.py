import csv
import gzip
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

from src.sequences import clean_sequence


DATASET_URL = "https://zenodo.org/records/3678563"
SOURCES = (
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
)
FIELDS = ["id", "sequence", "label", "source_file", "source_header", "source_split"]


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Calculate a file fingerprint without loading the file into memory.

    Args:
        path: File to read.
        algorithm: Name of the hash algorithm.

    Returns:
        A hexadecimal fingerprint.
    """
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    """Read ordinary or compressed FASTA records, including wrapped sequences.

    Args:
        path: A .fasta, .fa, or gzip-compressed file.

    Yields:
        The complete header and its sequence.

    Raises:
        ValueError: If records have missing headers or empty sequences.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    header = None
    letters = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    if not letters:
                        raise ValueError("Empty FASTA record: " + header)
                    yield header, "".join(letters)
                header = line[1:].strip()
                if not header:
                    raise ValueError("Missing FASTA header at line " + str(number))
                letters = []
            else:
                if header is None:
                    raise ValueError(
                        "Sequence before a FASTA header at line " + str(number)
                    )
                letters.append(line)
    if header is not None:
        if not letters:
            raise ValueError("Empty FASTA record: " + header)
        yield header, "".join(letters)


def download_source(source: dict, raw_dir: Path) -> Path:
    """Download and verify one published source file, reusing valid local copies.

    Args:
        source: Filename and expected MD5 from the dataset page.
        raw_dir: Directory for compressed source files.

    Returns:
        The verified local path.

    Raises:
        RuntimeError: If downloading or verification fails.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / source["filename"]
    if target.exists() and file_digest(target, "md5") == source["md5"]:
        print("Using verified local file: " + target.name, flush=True)
        return target
    url = DATASET_URL + "/files/" + target.name + "?download=1"
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            print("Downloading " + target.name, flush=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "dna-language-course-project/1.0"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                partial.open("wb") as output,
            ):
                transferred = 0
                progress = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    transferred += len(block)
                    if transferred - progress >= 20 * 1024 * 1024:
                        print(
                            "  received " + str(transferred // (1024 * 1024)) + " MB",
                            flush=True,
                        )
                        progress = transferred
            if file_digest(partial, "md5") != source["md5"]:
                raise RuntimeError(
                    "The downloaded file does not match its published fingerprint."
                )
            partial.replace(target)
            return target
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(
                    "Could not download "
                    + target.name
                    + ": "
                    + str(error)
                    + ". Check internet access, or place the original file in "
                    + str(raw_dir)
                ) from error
            time.sleep(1)
    raise RuntimeError("The download failed.")


def sample_records(
    path: Path, count: int, seed: int, label: str
) -> Tuple[List[dict], dict]:
    """Select a random candidate sample while reading every source record once.

    Args:
        path: A local FASTA source.
        count: Maximum number of candidate records to retain.
        seed: Fixed random setting.
        label: Label supplied by the source file.

    Returns:
        Candidate records and an audit of all records read.
    """
    rng = random.Random(seed)
    selected = []
    audit = Counter()
    for header, raw_sequence in read_fasta(path):
        audit["records_read"] += 1
        try:
            sequence = clean_sequence(raw_sequence, "dna")
        except ValueError:
            audit["rejected_invalid_letters"] += 1
            continue
        if len(sequence) != 250:
            audit["rejected_wrong_length"] += 1
            continue
        if set(sequence) - set("ACGT"):
            audit["records_with_ambiguous_letters"] += 1
        audit["eligible_records"] += 1
        seen = audit["eligible_records"]
        position = seen - 1 if seen <= count else rng.randrange(seen)
        if position < count:
            record = {
                "id": label + "_" + str(audit["records_read"]),
                "sequence": sequence,
                "label": label,
                "source_file": path.name,
                "source_header": header,
                "source_split": "validation",
            }
            if seen <= count:
                selected.append(record)
            else:
                selected[position] = record
        if audit["records_read"] % 250000 == 0:
            print(
                "  read " + str(audit["records_read"]) + " records from " + path.name,
                flush=True,
            )
    if not selected:
        raise ValueError("No usable 250-letter DNA records in " + str(path))
    return selected, dict(audit)


def curate_candidates(
    candidates: List[dict], per_class: int, seed: int
) -> Tuple[List[dict], dict]:
    """Remove repeated and conflicting sequences, then keep equal class sizes.

    Args:
        candidates: Random candidate records from both source files.
        per_class: Required number of final sequences in each class.
        seed: Fixed random setting.

    Returns:
        Shuffled unique records and duplicate counts within the candidate sample.

    Raises:
        ValueError: If a class lacks enough unique, unambiguous labels.
    """
    labels_by_sequence = {}
    for record in candidates:
        labels_by_sequence.setdefault(record["sequence"], set()).add(record["label"])
    conflicts = {
        sequence for sequence, labels in labels_by_sequence.items() if len(labels) > 1
    }
    grouped = {source["label"]: [] for source in SOURCES}
    seen = set()
    repeated = 0
    conflicting = 0
    for record in candidates:
        sequence = record["sequence"]
        if sequence in conflicts:
            conflicting += 1
        elif sequence in seen:
            repeated += 1
        else:
            grouped[record["label"]].append(record)
            seen.add(sequence)
    rng = random.Random(seed)
    result = []
    for label, records in grouped.items():
        if len(records) < per_class:
            raise ValueError(
                "Only "
                + str(len(records))
                + " unique candidates for "
                + label
                + "; request fewer sequences per class."
            )
        result.extend(rng.sample(records, per_class))
    rng.shuffle(result)
    return result, {
        "candidate_records": len(candidates),
        "repeated_candidate_records_removed": repeated,
        "conflicting_candidate_records_removed": conflicting,
        "final_class_counts": dict(Counter(record["label"] for record in result)),
    }


def prepare_dataset(
    data_dir: Path, per_class: int = 5000, seed: int = 42, force: bool = False
) -> Path:
    """Prepare a balanced, traceable sample of the DeePaC validation data.

    Args:
        data_dir: Parent directory for raw and processed data.
        per_class: Number of sequences to retain per class.
        seed: Fixed random setting for candidate sampling and final selection.
        force: Whether to replace an existing processed sample.

    Returns:
        The checked CSV path, reusing it if its settings and fingerprint match.

    Raises:
        ValueError: If settings or an existing sample are inconsistent.
    """
    if per_class < 1:
        raise ValueError("Sequences per class must be at least 1.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("The random setting must be between 0 and 4294967295.")
    processed = data_dir / "processed"
    target = processed / "sequences.csv"
    manifest_path = processed / "manifest.json"
    if target.exists() and not force:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("per_class") == per_class
                and manifest.get("seed") == seed
                and manifest.get("csv_sha256") == file_digest(target)
            ):
                print("Using the existing checked sample: " + str(target), flush=True)
                return target
        raise ValueError(
            "The existing sample has different settings or content. Use --force to replace it."
        )
    candidates = []
    sources = []
    for number, source in enumerate(SOURCES):
        path = download_source(source, data_dir / "raw")
        records, audit = sample_records(
            path, per_class * 2, seed + number, source["label"]
        )
        candidates.extend(records)
        sources.append(
            {
                **source,
                "url": DATASET_URL + "/files/" + source["filename"] + "?download=1",
                "sha256": file_digest(path),
                "audit": audit,
            }
        )
    records, audit = curate_candidates(candidates, per_class, seed)
    processed.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(target)
    manifest = {
        "dataset": "DeePaC IMG bacterial DNA reads",
        "dataset_url": DATASET_URL,
        "paper_url": "https://doi.org/10.1093/bioinformatics/btz541",
        "source_split": "validation",
        "alphabet": "dna",
        "per_class": per_class,
        "seed": seed,
        "sampling": "random reservoir candidates (2x requested), deduplication, balanced random selection",
        "sources": sources,
        "sample_audit": audit,
        "csv_sha256": file_digest(target),
        "label_meaning": "Labels belong to source organisms, not to individual disease-causing fragments.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared " + str(len(records)) + " sequences: " + str(target), flush=True)
    return target


def load_dataset(path: Path, alphabet: str = "dna") -> Tuple[List[dict], dict]:
    """Load a sequence CSV, check required values, and remove identical repeats.

    Args:
        path: CSV with id, sequence, and label columns.
        alphabet: Either dna or rna.

    Returns:
        Checked records and counts of duplicate rows removed.

    Raises:
        ValueError: If values are missing, IDs clash, or a sequence has conflicting labels.
    """
    records = []
    identifiers = {}
    seen = {}
    duplicates = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "sequence", "label"} <= set(reader.fieldnames or []):
            raise ValueError("The CSV needs id, sequence, and label columns.")
        for line, row in enumerate(reader, 2):
            if None in row or any(
                not (row.get(field) or "").strip()
                for field in ("id", "sequence", "label")
            ):
                raise ValueError("Missing or extra values on CSV line " + str(line))
            row = {key: value or "" for key, value in row.items()}
            row["id"] = row["id"].strip()
            row["label"] = row["label"].strip()
            try:
                row["sequence"] = clean_sequence(row["sequence"], alphabet)
            except ValueError as error:
                raise ValueError("CSV line " + str(line) + ": " + str(error)) from error
            signature = (row["sequence"], row["label"])
            if row["id"] in identifiers and identifiers[row["id"]] != signature:
                raise ValueError("Repeated sequence ID on CSV line " + str(line))
            identifiers[row["id"]] = signature
            previous = seen.get(row["sequence"])
            if previous is not None:
                if previous["label"] != row["label"]:
                    raise ValueError(
                        "The same sequence has conflicting labels on CSV line "
                        + str(line)
                    )
                duplicates += 1
                continue
            seen[row["sequence"]] = row
            records.append(row)
    if not records:
        raise ValueError("The dataset contains no sequences.")
    return records, {
        "input_rows": len(records) + duplicates,
        "duplicate_rows_removed": duplicates,
    }
