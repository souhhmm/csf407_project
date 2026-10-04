# Dataset source

The project uses the **IMG bacterial DNA reads supplied with DeePaC**, published as [Simulated NGS read datasets for bacterial pathogenic potential prediction](https://zenodo.org/records/3678563).

The reads were simulated from bacterial genomes downloaded from NCBI. Each read is 250 letters long. Labels come from the source organisms' metadata: `pathogenic` and `nonpathogenic`. They describe disease-causing potential of the source bacteria, rather than proving that a particular fragment causes disease.

## Files used

Only the two published validation files are downloaded. The published training and test files are not needed for steps 1–3.

| File | Label | Published MD5 fingerprint |
| --- | --- | --- |
| [nonpathogenic_val.fasta.gz](https://zenodo.org/records/3678563/files/nonpathogenic_val.fasta.gz?download=1) | nonpathogenic | `696379f88117ddd2f4512d078abde17f` |
| [pathogenic_val.fasta.gz](https://zenodo.org/records/3678563/files/pathogenic_val.fasta.gz?download=1) | pathogenic | `6b2e5c71634bd48a4c607e0979c09b2b` |

The compressed downloads occupy about 202 MB together. They are stored in `data/raw/`. Downloading is automatic through `run.py prepare`; if needed, original files placed in that directory are verified and reused.

The default prepared sample contains 5,000 unique sequences per class. Every eligible source record has the same chance of entering its class's random candidate sample. Deduplication and balanced selection happen after candidate sampling. This is a sample of reads, not a sample balanced by species.

`data/processed/manifest.json` records the actual fingerprints, source record counts, selection settings, and duplicate checks. `sequences.csv` preserves the source header, filename, class, and published split for each selected read.

The authors separated their original training, validation, and test data by bacterial species. Our exploratory sample comes from the validation files; it must not later be presented as an untouched final test set.

## Citation

Bartoszewicz, J. M., Seidel, A., Rentzsch, R., and Renard, B. Y. *Simulated NGS read datasets for bacterial pathogenic potential prediction*. Zenodo. [doi:10.5281/zenodo.3678563](https://doi.org/10.5281/zenodo.3678563).

Bartoszewicz, J. M., Seidel, A., Rentzsch, R., and Renard, B. Y. *DeePaC: predicting pathogenic potential of novel DNA with reverse-complement neural networks*. Bioinformatics, 36(1), 81–89. [doi:10.1093/bioinformatics/btz541](https://doi.org/10.1093/bioinformatics/btz541).
