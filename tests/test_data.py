import csv
import gzip
import hashlib
import io
import json

import pytest

from dna_language import data


def _write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "sequence", "label"])
        writer.writeheader()
        writer.writerows(records)


def _sequence(number):
    return "".join("ACGT"[(number // (4**digit)) % 4] for digit in range(8)) + "A" * 242


def test_plain_and_compressed_multiline_fasta(tmp_path):
    text = ">first full header\nacg\ntta\n\n>second\nGGCC\n"
    plain = tmp_path / "reads.fa"
    compressed = tmp_path / "reads.fa.gz"
    plain.write_text(text)
    with gzip.open(compressed, "wt") as handle:
        handle.write(text)
    expected = [("first full header", "acgtta"), ("second", "GGCC")]
    assert list(data.read_fasta(plain)) == expected
    assert list(data.read_fasta(compressed)) == expected


@pytest.mark.parametrize("text", ["ACGT\n", ">\nACGT\n", ">one\n>two\nACGT\n"])
def test_malformed_fasta_is_reported(tmp_path, text):
    path = tmp_path / "bad.fa"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(data.read_fasta(path))


def test_sampling_reads_the_full_file_and_repeats_with_the_same_seed(tmp_path):
    path = tmp_path / "reads.fa"
    text = "".join(
        ">read_" + str(number) + "\n" + _sequence(number) + "\n" for number in range(30)
    )
    path.write_text(text + ">short\nACGT\n>invalid\n" + "Z" * 250 + "\n")
    first, audit = data.sample_records(path, 5, 7, "pathogenic")
    second, _ = data.sample_records(path, 5, 7, "pathogenic")
    assert first == second
    assert len(first) == 5
    assert audit["records_read"] == 32
    assert audit["eligible_records"] == 30
    assert audit["rejected_wrong_length"] == 1
    assert audit["rejected_invalid_letters"] == 1
    assert any(int(record["source_header"].split("_")[-1]) >= 5 for record in first)


def test_curation_removes_duplicate_and_conflicting_candidates():
    candidates = [
        {"sequence": "AAAA", "label": "pathogenic"},
        {"sequence": "AAAA", "label": "pathogenic"},
        {"sequence": "CCCC", "label": "pathogenic"},
        {"sequence": "GGGG", "label": "nonpathogenic"},
        {"sequence": "TTTT", "label": "nonpathogenic"},
        {"sequence": "ACGT", "label": "pathogenic"},
        {"sequence": "ACGT", "label": "nonpathogenic"},
    ]
    records, audit = data.curate_candidates(candidates, 2, 42)
    assert len(records) == 4
    assert len({record["sequence"] for record in records}) == 4
    assert audit["repeated_candidate_records_removed"] == 1
    assert audit["conflicting_candidate_records_removed"] == 2
    assert audit["final_class_counts"] == {"pathogenic": 2, "nonpathogenic": 2}


def test_download_checks_the_fingerprint_and_reuses_a_local_file(tmp_path, monkeypatch):
    payload = b"test download contents"
    source = {"filename": "reads.fa.gz", "md5": hashlib.md5(payload).hexdigest()}
    calls = []

    def response(*args, **kwargs):
        calls.append(args)
        return io.BytesIO(payload)

    monkeypatch.setattr(data.urllib.request, "urlopen", response)
    path = data.download_source(source, tmp_path)
    assert path.read_bytes() == payload
    assert data.download_source(source, tmp_path) == path
    assert len(calls) == 1


def test_corrupt_download_does_not_replace_an_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "reads.fa.gz"
    target.write_bytes(b"original")
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt")
    )
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="fingerprint"):
        data.download_source({"filename": target.name, "md5": "0" * 32}, tmp_path)
    assert target.read_bytes() == b"original"
    assert not (tmp_path / "reads.fa.gz.part").exists()


def test_preparation_is_balanced_traceable_and_detects_modified_samples(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    raw.mkdir()
    sources = []
    for number, label in enumerate(("nonpathogenic", "pathogenic")):
        path = raw / (label + ".fa.gz")
        with gzip.open(path, "wt") as handle:
            for index in range(15):
                handle.write(
                    ">read_"
                    + str(index)
                    + "\n"
                    + _sequence(index + number * 100)
                    + "\n"
                )
        sources.append(
            {
                "filename": path.name,
                "label": label,
                "md5": data.file_digest(path, "md5"),
            }
        )
    monkeypatch.setattr(data, "SOURCES", tuple(sources))
    path = data.prepare_dataset(tmp_path, per_class=3, seed=4)
    records, _ = data.load_dataset(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert len(records) == 6
    assert manifest["sample_audit"]["final_class_counts"] == {
        "pathogenic": 3,
        "nonpathogenic": 3,
    }
    assert manifest["csv_sha256"] == data.file_digest(path)
    assert all(record["source_header"] for record in records)
    assert data.prepare_dataset(tmp_path, 3, 4) == path
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="different settings or content"):
        data.prepare_dataset(tmp_path, 3, 4)


def test_csv_duplicates_are_checked_after_normalization(tmp_path):
    path = tmp_path / "sequences.csv"
    _write_csv(
        path,
        [
            {"id": "first", "sequence": "acgt", "label": "a"},
            {"id": "second", "sequence": "AC GT", "label": "a"},
        ],
    )
    records, audit = data.load_dataset(path)
    assert len(records) == 1
    assert audit["duplicate_rows_removed"] == 1


def test_ids_of_removed_duplicate_rows_cannot_be_reused_for_other_sequences(tmp_path):
    path = tmp_path / "sequences.csv"
    _write_csv(
        path,
        [
            {"id": "first", "sequence": "ACGT", "label": "a"},
            {"id": "second", "sequence": "ACGT", "label": "a"},
            {"id": "second", "sequence": "GGGG", "label": "a"},
        ],
    )
    with pytest.raises(ValueError, match="Repeated sequence ID"):
        data.load_dataset(path)


def test_conflicting_labels_are_reported(tmp_path):
    path = tmp_path / "sequences.csv"
    _write_csv(
        path,
        [
            {"id": "first", "sequence": "ACGT", "label": "a"},
            {"id": "second", "sequence": "ACGT", "label": "b"},
        ],
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        data.load_dataset(path)


@pytest.mark.parametrize(
    "text",
    [
        "sequence,label\nACGT,a\n",
        "id,sequence,label\none,ACGT,\n",
        "id,sequence,label\none,ACGT,a,extra\n",
    ],
)
def test_missing_or_extra_csv_values_are_reported(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        data.load_dataset(path)
