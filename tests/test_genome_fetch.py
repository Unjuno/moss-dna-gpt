import gzip

from moss_dna_gpt.genome_fetch import parse_assembly_summary, select_assembly, safe_name, verify_gzip_readable


def test_parse_assembly_summary_and_selects_latest_full_scaffold():
    text = """# comment\n# assembly_accession\tbioproject\tbiosample\twgs_master\trefseq_category\ttaxid\tspecies_taxid\torganism_name\tinfraspecific_name\tisolate\tversion_status\tassembly_level\trelease_type\tgenome_rep\tseq_rel_date\tasm_name\tsubmitter\tgbrs_paired_asm\tpaired_asm_comp\tftp_path\texcluded_from_refseq\trelation_to_type_material\nGCA_OLD\tna\tna\tna\tna\t3218\t3218\tPhyscomitrium patens\tna\tna\told\tContig\tMinor\tFull\t2020-01-01\told\tna\tna\tna\thttps://example.invalid/old\tna\tna\nGCA_NEW\tna\tna\tna\tna\t3218\t3218\tPhyscomitrium patens\tna\tna\tlatest\tScaffold\tMajor\tFull\t2024-01-01\tnew\tna\tna\tna\thttps://example.invalid/new\tna\tna\n"""
    rows = parse_assembly_summary(text)
    selected = select_assembly(rows, taxid="3218")
    assert len(rows) == 2
    assert selected.accession == "GCA_NEW"
    assert selected.fasta_url == "https://example.invalid/new/new_genomic.fna.gz"


def test_safe_name():
    assert safe_name("Physcomitrium patens") == "physcomitrium_patens"


def test_verify_gzip_readable(tmp_path):
    path = tmp_path / 'ok.fna.gz'
    with gzip.open(path, 'wb') as fp:
        fp.write(b'>x\nACGT\n')
    result = verify_gzip_readable(path)
    assert result['verified_readable'] is True
    assert result['sample_bytes'] == 1


def test_verify_gzip_readable_rejects_plain_text(tmp_path):
    path = tmp_path / 'bad.fna.gz'
    path.write_text('not gzip', encoding='utf-8')
    result = verify_gzip_readable(path)
    assert result['verified_readable'] is False
    assert result['error']
