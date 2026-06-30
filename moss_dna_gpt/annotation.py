from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SEQID_TRANSLATE = {
    "NC_037253.1": "Chr01",
    "NC_037254.1": "Chr02",
    "NC_037255.1": "Chr03",
    "NC_037256.1": "Chr04",
    "NC_037257.1": "Chr05",
    "NC_037258.1": "Chr06",
    "NC_037259.1": "Chr07",
    "NC_037260.1": "Chr08",
    "NC_037261.1": "Chr09",
    "NC_037262.1": "Chr10",
    "NC_037263.1": "Chr11",
    "NC_037264.1": "Chr12",
    "NC_037265.1": "Chr13",
    "NC_037266.1": "Chr14",
    "NC_037267.1": "Chr15",
    "NC_037268.1": "Chr16",
    "NC_037269.1": "Chr17",
    "NC_037270.1": "Chr18",
    "NC_037271.1": "Chr19",
    "NC_037272.1": "Chr20",
    "NC_037273.1": "Chr21",
    "NC_037274.1": "Chr22",
    "NC_037275.1": "Chr23",
    "NC_037276.1": "Chr24",
    "NC_037277.1": "Chr25",
    "NC_037278.1": "Chr26",
    "NC_037279.1": "Chr27",
}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

PRIORITY = [
    "CDS",
    "five_prime_UTR",
    "three_prime_UTR",
    "UTR",
    "exon",
    "mRNA",
    "gene",
    "ncRNA",
]


@dataclass
class GffFeature:
    seqid: str
    source: str
    type: str
    start: int
    end: int
    score: str
    strand: str
    phase: str
    attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_line(cls, line: str, translate_seqid: bool = True) -> GffFeature:
        parts = line.strip().split("\t")
        if len(parts) != 9:
            raise ValueError(f"invalid GFF line: {line!r}")
        seqid = parts[0]
        if translate_seqid:
            seqid = SEQID_TRANSLATE.get(seqid, seqid)
        attrs = {}
        for pair in parts[8].split(";"):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                attrs[k] = v
        try:
            start = int(parts[3])
            end = int(parts[4])
        except ValueError as e:
            raise ValueError(f"invalid GFF integer field in line: {line!r}") from e
        return cls(
            seqid=seqid,
            source=parts[1],
            type=parts[2],
            start=start,
            end=end,
            score=parts[5],
            strand=parts[6],
            phase=parts[7],
            attributes=attrs,
        )


def parse_gff(path: str | Path, translate_seqid: bool = True) -> list[GffFeature]:
    features = []
    opener = gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)
    with opener as fp:
        for line in fp:
            if line.startswith("#") or not line.strip():
                continue
            features.append(GffFeature.from_line(line, translate_seqid=translate_seqid))
    return features


def _complement(seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    return "".join(comp.get(c, "N") for c in seq)


def extract_codon(
    seq: str,
    cds_start: int,
    cds_strand: str,
    phase: int,
) -> tuple[str, int]:
    offset = phase
    if cds_strand == "+":
        codon_start = cds_start - 1 + offset
        codon = seq[codon_start : codon_start + 3]
    else:
        rev = _complement(seq[::-1])
        codon_start = offset
        codon = rev[codon_start : codon_start + 3]
    return codon.upper()


def classify_variant(
    features: list[GffFeature],
    seqid: str,
    pos: int,
    ref_seq: str | None = None,
) -> dict:
    hits = []
    for f in features:
        if f.seqid != seqid:
            continue
        if f.start <= pos <= f.end:
            hits.append(f)

    best_type: str = "intergenic"
    best_feature: GffFeature | None = None

    for ptype in PRIORITY:
        for f in hits:
            if f.type == ptype:
                best_type = ptype
                best_feature = f
                break
        if best_feature is not None:
            break

    if best_feature is not None:
        for f in hits:
            if f.type == "gene" and best_feature.type in ("CDS", "exon", "UTR",
                                                          "five_prime_UTR",
                                                          "three_prime_UTR",
                                                          "mRNA"):
                if best_feature.attributes.get("Parent") in (
                    f.attributes.get("ID"),
                    None,
                ):
                    pass

    result: dict = {
        "region_type": best_type,
        "feature_id": best_feature.attributes.get("ID") if best_feature else None,
        "gene_id": None,
        "strand": best_feature.strand if best_feature else None,
        "synonymous": None,
    }

    for f in hits:
        if f.type == "gene":
            result["gene_id"] = f.attributes.get("ID")
            break
    if result["gene_id"] is None and best_feature is not None:
        for f in hits:
            if f.type == "gene":
                result["gene_id"] = f.attributes.get("ID")
                break

    if best_type == "CDS" and best_feature is not None and ref_seq is not None:
        phase = int(best_feature.phase) if best_feature.phase.isdigit() else 0
        ref_codon = extract_codon(ref_seq, best_feature.start, best_feature.strand, phase)
        if len(ref_codon) == 3:
            codon_pos = ((pos - best_feature.start) - int(best_feature.phase)) % 3
            alt_codon = list(ref_codon)
            alt_codon[codon_pos] = result.get("alt_base", "N")
            alt_codon_str = "".join(alt_codon)
            ref_aa = CODON_TABLE.get(ref_codon.upper(), "X")
            alt_aa = CODON_TABLE.get(alt_codon_str.upper(), "X")
            result["synonymous"] = ref_aa == alt_aa
            result["ref_codon"] = ref_codon
            result["alt_codon"] = alt_codon_str
            result["ref_aa"] = ref_aa
            result["alt_aa"] = alt_aa

    return result
