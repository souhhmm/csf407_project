import pytest

from src.data import read_fasta
from src.splits import (
    audit_splits,
    genome_from_header,
    load_splits,
    prepare_splits,
    sample_split_records,
)


SIZES = {"train": 100, "val": 40, "test_1": 40, "test_2": 40}


def test_genome_is_read_from_deepac_header():
    header = "/home/x/trainingReads/pathogenic/GCA_000003215.1.fq.000000007 "
    assert genome_from_header(header) == "GCA_000003215.1"
    assert genome_from_header("plain_name") == "plain_name"


def test_sampler_rejects_short_reads_and_keeps_requested_count(synthetic_sources):
    data_dir, sources = synthetic_sources
    path = data_dir / "raw" / sources["val"][0]["filename"]
    records, audit = sample_split_records(path, 50, 1, "nonpathogenic", "val")
    assert len(records) == 50
    assert audit["rejected_wrong_length_or_letters"] == 1
    assert audit["eligible_records"] == 160
    assert {record["source_split"] for record in records} == {"val"}
    assert all(len(record["sequence"]) == 250 for record in records)
    assert len(list(read_fasta(path))) == 161


def test_prepared_splits_are_balanced_unique_and_genome_disjoint(
    synthetic_sources, tmp_path
):
    data_dir, sources = synthetic_sources
    splits_dir = tmp_path / "splits"
    prepare_splits(data_dir, splits_dir, SIZES, 7, sources=sources)
    records, manifest = load_splits(splits_dir)
    assert {name: len(rows) for name, rows in records.items()} == {
        name: 2 * count for name, count in SIZES.items()
    }
    for name, rows in records.items():
        labels = [row["label"] for row in rows]
        assert (
            labels.count("pathogenic") == labels.count("nonpathogenic") == SIZES[name]
        )
        assert len({row["sequence"] for row in rows}) == len(rows)
    assert not (
        {row["genome"] for row in records["train"]}
        & {row["genome"] for row in records["test_1"]}
    )
    assert manifest["split_audit"]["genome_overlaps"]["train & val"] == 0
    assert "test_1 & test_2" not in manifest["split_audit"]["genome_overlaps"]


def test_existing_splits_are_reused_only_when_settings_match(
    synthetic_sources, tmp_path
):
    data_dir, sources = synthetic_sources
    splits_dir = tmp_path / "splits"
    prepare_splits(data_dir, splits_dir, SIZES, 7, sources=sources)
    before = (splits_dir / "train.csv").read_bytes()
    prepare_splits(data_dir, splits_dir, SIZES, 7, sources=sources)
    assert (splits_dir / "train.csv").read_bytes() == before
    with pytest.raises(ValueError, match="different settings"):
        prepare_splits(data_dir, splits_dir, {**SIZES, "train": 50}, 7, sources=sources)
    (splits_dir / "val.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed since"):
        load_splits(splits_dir)


def test_shared_genomes_between_splits_are_refused():
    records = {
        "train": [{"sequence": "AAAA", "genome": "g1"}],
        "val": [{"sequence": "CCCC", "genome": "g1"}],
    }
    with pytest.raises(ValueError, match="share genomes"):
        audit_splits(records)
    records["val"][0]["genome"] = "g2"
    assert audit_splits(records)["genome_overlaps"] == {"train & val": 0}


def test_mate_pairing_of_the_two_test_sets_is_recorded(synthetic_sources, tmp_path):
    data_dir, sources = synthetic_sources
    splits_dir = tmp_path / "splits"
    prepare_splits(data_dir, splits_dir, SIZES, 7, sources=sources)
    _, manifest = load_splits(splits_dir)
    paired = manifest["split_audit"]["paired_test_reads"]
    assert paired["test_1_reads"] == paired["test_2_reads"] == 2 * SIZES["test_1"]
    assert paired["reads_in_both"] == 0
