import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


import gzip
import hashlib

import numpy as np
import pytest


def _reads(rng, count, gc, per_genome, tag):
    reads = []
    for number in range(count):
        letters = rng.choice(
            list("ACGT"), size=250, p=[(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        )
        if number % 7 == 0:
            letters[100] = "N"
        genome = "GCA_" + tag + str(number // per_genome) + ".1"
        header = "/data/reads/" + genome + ".fq." + format(number, "09d") + " "
        reads.append((header, "".join(letters)))
    return reads


@pytest.fixture
def synthetic_sources(tmp_path):
    """Write tiny gzip FASTA files for every split and return their descriptions.

    Pathogenic reads are G/C-rich and nonpathogenic reads are A/T-rich, so a
    composition model should separate them, and every split uses new genomes.
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    sources = {}
    for split, count in (
        ("train", 400),
        ("val", 160),
        ("test_1", 160),
        ("test_2", 160),
    ):
        sources[split] = []
        for label, gc in (("nonpathogenic", 0.40), ("pathogenic", 0.55)):
            name = label + "_" + split + ".fasta.gz"
            reads = _reads(rng, count, gc, 10, split + label[:2])
            with gzip.open(raw / name, "wt") as handle:
                for header, sequence in reads:
                    handle.write(">" + header + "\n" + sequence + "\n")
            reads_short = ">short\nACGT\n"
            with gzip.open(raw / name, "at") as handle:
                handle.write(reads_short)
            sources[split].append(
                {
                    "filename": name,
                    "label": label,
                    "md5": hashlib.md5((raw / name).read_bytes()).hexdigest(),
                }
            )
    return tmp_path / "data", {key: tuple(value) for key, value in sources.items()}
