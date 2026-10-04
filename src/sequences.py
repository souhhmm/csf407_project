from collections import Counter
from typing import List


ALPHABETS = {"dna": "ACGT", "rna": "ACGU"}
AMBIGUOUS = set("RYSWKMBDHVN")


def clean_sequence(sequence: str, alphabet: str = "dna") -> str:
    """Normalize a sequence without guessing unknown letters.

    Args:
        sequence: Letters to check; spacing and letter case are ignored.
        alphabet: Either dna or rna. T and U are not silently exchanged.

    Returns:
        The checked, uppercase sequence, retaining ambiguous letters.

    Raises:
        ValueError: If the alphabet, sequence, or a letter is invalid.
    """
    if alphabet not in ALPHABETS:
        raise ValueError("Choose dna or rna as the alphabet.")
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("The sequence is empty.")
    unexpected = set(cleaned) - set(ALPHABETS[alphabet]) - AMBIGUOUS
    if unexpected:
        raise ValueError(
            "Unexpected sequence letters: " + ", ".join(sorted(unexpected))
        )
    return cleaned


def make_words(sequence: str, k: int, alphabet: str = "dna") -> List[str]:
    """Make overlapping words, skipping windows containing unknown letters.

    Args:
        sequence: A DNA or RNA sequence.
        k: Number of letters in each word.
        alphabet: Either dna or rna.

    Returns:
        Words in their original order. No word crosses an ambiguous letter.

    Raises:
        ValueError: If k or the sequence is invalid.
    """
    if k < 1:
        raise ValueError("Word size must be at least 1.")
    sequence = clean_sequence(sequence, alphabet)
    allowed = set(ALPHABETS[alphabet])
    return [
        sequence[start : start + k]
        for start in range(len(sequence) - k + 1)
        if set(sequence[start : start + k]) <= allowed
    ]


def word_segments(sequence: str, k: int, alphabet: str = "dna") -> List[List[str]]:
    """Keep words around unknown letters in separate context groups.

    Args:
        sequence: A DNA or RNA sequence.
        k: Number of letters in each word.
        alphabet: Either dna or rna.

    Returns:
        Nonempty word lists from uninterrupted runs of known letters.

    Raises:
        ValueError: If k or the sequence is invalid.
    """
    if k < 1:
        raise ValueError("Word size must be at least 1.")
    sequence = clean_sequence(sequence, alphabet)
    segments = []
    run = []
    for letter in sequence + "N":
        if letter in ALPHABETS[alphabet]:
            run.append(letter)
        else:
            if len(run) >= k:
                segments.append(make_words("".join(run), k, alphabet))
            run = []
    return segments


def sequence_statistics(sequence: str, alphabet: str = "dna") -> dict:
    """Calculate length and letter proportions for a checked sequence.

    Args:
        sequence: A DNA or RNA sequence.
        alphabet: Either dna or rna.

    Returns:
        Length, ambiguous-letter count, known-letter proportions, and GC fraction.
        Proportions are unavailable (NaN) when no letters are known.
    """
    sequence = clean_sequence(sequence, alphabet)
    counts = Counter(sequence)
    known = sum(counts[letter] for letter in ALPHABETS[alphabet])
    result = {"length": len(sequence), "ambiguous_letters": len(sequence) - known}
    result["gc_fraction"] = (
        (counts["G"] + counts["C"]) / known if known else float("nan")
    )
    for letter in ALPHABETS[alphabet]:
        result[letter + "_fraction"] = counts[letter] / known if known else float("nan")
    return result
