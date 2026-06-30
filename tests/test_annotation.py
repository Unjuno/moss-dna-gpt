import pytest

from moss_dna_gpt.annotation import GffFeature, parse_gff, classify_variant


GFF_SAMPLE = """##gff-version 3
NC_037253.1\tGnomon\tgene\t1000\t5000\t.\t+\t.\tID=gene1;Name=SAMPLE
NC_037253.1\tGnomon\tmRNA\t1000\t5000\t.\t+\t.\tID=rna1;Parent=gene1
NC_037253.1\tGnomon\texon\t1000\t1500\t.\t+\t.\tID=exon1;Parent=rna1
NC_037253.1\tGnomon\tCDS\t1000\t1200\t.\t+\t0\tID=cds1;Parent=rna1
NC_037253.1\tGnomon\tfive_prime_UTR\t1201\t1500\t.\t+\t.\tID=utr5;Parent=rna1
NC_037253.1\tGnomon\tCDS\t2000\t5000\t.\t+\t1\tID=cds2;Parent=rna1
"""


def test_gff_feature_from_line_parses_columns():
    line = "NC_037253.1\tGnomon\tgene\t1000\t5000\t.\t+\t.\tID=gene1;Name=SAMPLE"
    f = GffFeature.from_line(line)
    assert f.seqid == "Chr01"
    assert f.source == "Gnomon"
    assert f.type == "gene"
    assert f.start == 1000
    assert f.end == 5000
    assert f.score == "."
    assert f.strand == "+"
    assert f.phase == "."
    assert f.attributes["ID"] == "gene1"
    assert f.attributes["Name"] == "SAMPLE"


def test_parse_gff_skips_comments_and_headers(tmp_path):
    p = tmp_path / "test_gff_skip.gff"
    p.write_text(GFF_SAMPLE)
    result = parse_gff(p)
    assert len(result) == 6
    assert all(isinstance(f, GffFeature) for f in result)


def test_classify_variant_intergenic(tmp_path):
    p = tmp_path / "test_gff_classify.gff"
    p.write_text(GFF_SAMPLE)
    features = parse_gff(p)
    result = classify_variant(features, "Chr01", 1)
    assert result["region_type"] == "intergenic"


def test_classify_variant_cds(tmp_path):
    p = tmp_path / "test_gff_cds.gff"
    p.write_text(GFF_SAMPLE)
    features = parse_gff(p)
    result = classify_variant(features, "Chr01", 1100)
    assert result["region_type"] == "CDS"
    assert result["gene_id"] == "gene1"


def test_classify_variant_utr(tmp_path):
    p = tmp_path / "test_gff_utr.gff"
    p.write_text(GFF_SAMPLE)
    features = parse_gff(p)
    result = classify_variant(features, "Chr01", 1300)
    assert result["region_type"] == "five_prime_UTR"
