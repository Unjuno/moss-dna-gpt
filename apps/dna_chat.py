from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.sampling import sample
from moss_dna_gpt.scoring import score_variant
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.trainer import load_checkpoint, resolve_device
from moss_dna_gpt.evaluate import (
    evaluate_biological_quality,
    gc_content,
    gc_distribution,
    kmer_frequencies,
    sample_windows_from_fasta,
    sequence_complexity,
    shannon_entropy,
    cpg_obs_exp,
    pyrimidine_dimer_score,
)
from moss_dna_gpt.embed import (
    extract_embeddings,
    build_similarity_graph,
    build_debruijn_graph,
    compute_umap,
    compute_baseline_stats,
)
from moss_dna_gpt.annotation_utils import (
    load_annotation_db,
    annotate_sequences_from_fasta,
)

DNA_RE = re.compile(r'[^ACGTNacgtn]+')
DEFAULT_CHECKPOINT = 'runs/physcomitrium_patens_quick_256/ckpt_step_50.pt'
REAL_FASTA = 'data/raw/physcomitrium_patens_manual/GCA_000002425.3_Phypa_V5_genomic.fna.gz'
GFF_PATH = 'data/annotations/Physcomitrium_patens.Phypa_V3.63.gff3.gz'
_EVAL_CURVE_PATH = Path(__file__).resolve().parents[1] / 'repo' / 'moss-dna-gpt-20m-patens' / 'eval_curve.json'

GEN_PRESETS: dict[str, str] = {
    'Custom': '',
    'AT-rich region (promoter-like)': 'TATAAATAGCTAGCTAGCTAGCTAGCTAGCTAGC',
    'GC-rich region (coding-like)': 'GCCGCCATGGCCGAGCTCGAGCTCGAGCTCGAG',
    'Simple repeat': 'ACGTACGTACGTACGTACGTACGTACGTACGT',
    'Start codon context': 'GCCGCCACCATGGCCGAGCTCGAGCTCGAG',
    'N-rich (low complexity)': 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN',
}

SCORE_PRESETS: dict[str, dict] = {
    'Custom': {'seq': '', 'pos': 0, 'alt': 'G'},
    'SNP in AT-rich region': {'seq': 'TATAAATAGCTAGCTAGCTAGC', 'pos': 5, 'alt': 'G'},
    'SNP in GC-rich region': {'seq': 'GCCGCCATGGCCGAGCTCGAG', 'pos': 5, 'alt': 'T'},
    'SNP at repeat start': {'seq': 'ACGTACGTACGTACGTACGT', 'pos': 0, 'alt': 'T'},
    'Transversion in coding-like': {'seq': 'ATGGCCGAGCTCGAGCTCGAG', 'pos': 3, 'alt': 'C'},
}


@st.cache_data
def _load_eval_data():
    default = {"dna_gpt_curve": [], "markov_baselines": {}}
    if _EVAL_CURVE_PATH.exists():
        with open(_EVAL_CURVE_PATH) as fp:
            return json.load(fp)
    return default


def build_curve_df():
    data = _load_eval_data()
    curve = data.get("dna_gpt_curve", [])
    markov = data.get("markov_baselines", {})
    if not curve:
        return pd.DataFrame()
    rows = []
    for pt in curve:
        row = {"step": pt["step"], "DNA-GPT": pt["bits_per_base"]}
        for name, v in markov.items():
            row[name] = v["bits_per_base"]
        rows.append(row)
    return pd.DataFrame(rows)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--config', default=None, help='Path to config.json (required for .safetensors checkpoints)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--fasta', default=REAL_FASTA)
    args, _ = parser.parse_known_args()
    return args


def clean_dna(text: str) -> str:
    return DNA_RE.sub('', text).upper()


def base_stats(seq: str) -> dict[str, float | int]:
    total = len(seq)
    stats: dict[str, float | int] = {'length': total}
    for base in 'ACGTN':
        count = seq.count(base)
        stats[f'{base}_count'] = count
        stats[f'{base}_ratio'] = count / total if total else 0.0
    return stats


def render_base_chart(stats: dict) -> None:
    counts = {b: stats[f'{b}_count'] for b in 'ACGTN'}
    df = pd.DataFrame([counts])
    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(df, height=200)
    with col2:
        gc = (stats.get('G_count', 0) + stats.get('C_count', 0)) / max(stats.get('length', 1), 1)
        st.metric('Length', stats.get('length', 0))
        st.metric('GC content', f'{gc:.1%}')
        for b in 'ACGTN':
            st.metric(f'{b}', f'{stats.get(f"{b}_ratio", 0):.1%}')


@st.cache_resource(show_spinner='Loading checkpoint...')
def load_model(checkpoint_path: str, device: str, config_path: str | None = None):
    selected = resolve_device(device)
    tokenizer = DnaTokenizer()
    if checkpoint_path.endswith('.safetensors'):
        config_file = Path(config_path) if config_path else Path(checkpoint_path).parent / 'config.json'
        with open(config_file) as f:
            cfg = json.load(f)
        config = GPTConfig(**{k: cfg[k] for k in ['vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias'] if k in cfg})
        model = GPT(config)
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
        model.to(selected)
        model.eval()
        meta = {'step': None, 'model_config': cfg}
        return model, meta, tokenizer, selected
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=selected)
    model.to(selected)
    return model, checkpoint, tokenizer, selected


def generate_continuation(checkpoint_path, prefix, max_new_tokens, temperature, top_k, device, config_path=None):
    model, checkpoint, tokenizer, selected = load_model(checkpoint_path, device, config_path)
    cleaned = clean_dna(prefix)
    if not cleaned:
        raise ValueError('Prefix must contain at least one A/C/G/T/N base.')
    ids = tokenizer.encode(cleaned, unknown='n')
    idx = torch.tensor([ids], dtype=torch.long, device=selected)
    allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
    with torch.no_grad():
        out = sample(model, idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, allowed_token_ids=allowed)
    full = tokenizer.decode(out[0].tolist(), skip_special=True)
    return full, base_stats(full), selected, checkpoint.get('step'), model.num_parameters()


def generate_sliding_window(checkpoint_path, prefix, total_tokens, chunk_size, temperature, top_k, device, config_path=None, progress_callback=None, text_callback=None):
    model, checkpoint, tokenizer, selected = load_model(checkpoint_path, device, config_path)
    cleaned = clean_dna(prefix)
    if not cleaned:
        raise ValueError('Prefix must contain at least one A/C/G/T/N base.')
    ids = tokenizer.encode(cleaned, unknown='n')
    idx = torch.tensor([ids], dtype=torch.long, device=selected)
    allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
    block_size = model.config.block_size
    chunks = (total_tokens + chunk_size - 1) // chunk_size
    with torch.no_grad():
        for c in range(chunks):
            remain = total_tokens - c * chunk_size
            this_chunk = min(chunk_size, remain)
            for _ in range(this_chunk):
                logits, _ = model(idx[:, -block_size:])
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                if allowed is not None:
                    mask = torch.full_like(logits, float('-inf'))
                    mask[:, allowed] = logits[:, allowed]
                    logits = mask
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1, None]] = float('-inf')
                idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
            if text_callback:
                text_callback(tokenizer.decode(idx[0].tolist(), skip_special=True))
            if progress_callback:
                progress_callback((c + 1) / chunks)
    full = tokenizer.decode(idx[0].tolist(), skip_special=True)
    return full, base_stats(full), selected, checkpoint.get('step'), model.num_parameters()


def render_learning_curve():
    with st.expander('Model performance vs Markov baselines', expanded=False):
        data = _load_eval_data()
        df = build_curve_df()
        if not df.empty:
            markov = data.get("markov_baselines", {})
            columns = ['DNA-GPT'] + [k for k in markov.keys()]
            st.line_chart(df, x='step', y=columns, height=350, width='stretch')
            curve = data.get("dna_gpt_curve", [])
            final_bits = curve[-1]["bits_per_base"] if curve else 0
            cols = st.columns(len(markov) + 1)
            cols[0].metric('DNA-GPT (8M steps)', f'{final_bits:.4f}')
            for i, (name, v) in enumerate(markov.items()):
                cols[i + 1].metric(name, f'{v["bits_per_base"]:.4f}')
            improvement = ((list(markov.values())[-1]["bits_per_base"] - final_bits) / list(markov.values())[-1]["bits_per_base"]) * 100 if markov else 0
            st.success(f'DNA-GPT outperforms all Markov baselines by **{improvement:.1f}%** (bits/base)')
        else:
            st.info('Run `python scripts/eval_all_checkpoints.py` to generate eval data.')


def render_generate_tab(checkpoint, device, temperature, top_k, sliding, total_tokens, chunk_size, max_new_tokens, cli, config_path):
    if 'history' not in st.session_state:
        st.session_state.history = []
    use_case = st.selectbox('Use case', list(GEN_PRESETS.keys()), index=0, key='gen_usecase')
    if use_case == 'Custom':
        prefix = st.text_area('DNA prefix', value='ACGTACGTACGT', height=120, key='dna_prefix_input', help='Only A/C/G/T/N are kept.')
    else:
        prefix = st.text_area('DNA prefix', value=GEN_PRESETS[use_case], height=80, key='dna_prefix_input', help='Only A/C/G/T/N are kept.')
        st.caption(f'**{use_case}** — the model will continue this sequence based on patterns learned from the moss genome.')
    col_a, col_b = st.columns([1, 1])
    with col_a:
        run = st.button('Generate continuation', type='primary')
    with col_b:
        clear = st.button('Clear history')
    if clear:
        st.session_state.history = []
    if run:
        try:
            used_prefix = st.session_state.dna_prefix_input
            progress_bar = st.progress(0.0, text='Generating...')
            if sliding:
                output_placeholder = st.empty()
                full, stats, selected_device, step, param_count = generate_sliding_window(checkpoint, used_prefix, total_tokens=total_tokens, chunk_size=chunk_size, temperature=temperature, top_k=top_k, device=device, config_path=config_path, progress_callback=lambda v: progress_bar.progress(v, text=f'Generating... {v*100:.0f}%'), text_callback=lambda t: output_placeholder.markdown(f'<div style="font-family: monospace; white-space: pre-wrap; word-break: break-word; height: 400px; overflow-y: auto; background: #f0f0f0; padding: 10px; border-radius: 4px;">{t}</div>', unsafe_allow_html=True))
            else:
                full, stats, selected_device, step, param_count = generate_continuation(checkpoint, used_prefix, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, device=device, config_path=config_path)
            progress_bar.empty()
            if sliding:
                output_placeholder.empty()
            item = {'prefix': clean_dna(used_prefix), 'output': full, 'stats': stats, 'device': selected_device, 'step': step, 'param_count': param_count, 'temperature': temperature, 'top_k': top_k}
            st.session_state.history.insert(0, item)
        except Exception as exc:
            st.error(str(exc))
    for i, item in enumerate(st.session_state.history):
        with st.container(border=True):
            st.subheader(f'Generation {i + 1}')
            st.write(f"device: `{item['device']}`  step: `{item['step']}`  params: `{item['param_count']}`  temperature: `{item['temperature']}`  top_k: `{item['top_k']}`")
            st.markdown(f'<div style="font-family: monospace; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; background: #f0f0f0; padding: 10px; border-radius: 4px; font-size: 0.85em;">{item["output"]}</div>', unsafe_allow_html=True)
            render_base_chart(item['stats'])


def render_variant_tab(checkpoint, device, config_path=None):
    st.subheader('Variant Effect Scoring')
    st.caption('Measure how surprising a single-nucleotide variant is to the DNA language model.')
    score_case = st.selectbox('Use case', list(SCORE_PRESETS.keys()), index=0, key='score_usecase')
    preset = SCORE_PRESETS[score_case]
    if score_case == 'Custom':
        ref_seq = st.text_area('Reference sequence', value='ACGTACGTACGTACGTACGTACGTACGT', height=120, key='score_seq', help='Wild-type DNA sequence.')
    else:
        ref_seq = st.text_area('Reference sequence', value=preset['seq'], height=80, key='score_seq', help='Wild-type DNA sequence.')
        st.caption(f'**{score_case}** — scoring a {preset["alt"]} substitution at position {preset["pos"]}.')
    col_p, col_a = st.columns([1, 1])
    with col_p:
        pos = st.number_input('Position (0-indexed)', min_value=0, max_value=10000, value=preset['pos'], step=1, key='score_pos')
    with col_a:
        alt = st.selectbox('Alternate base', ['A', 'C', 'G', 'T', 'N'], index=['A', 'C', 'G', 'T', 'N'].index(preset['alt']), key='score_alt')
    if st.button('Score variant', type='primary', key='score_btn'):
        try:
            cleaned = clean_dna(ref_seq)
            if not cleaned:
                raise ValueError('Sequence must contain at least one A/C/G/T/N base.')
            if pos >= len(cleaned):
                raise ValueError(f'Position {pos} out of range for sequence of length {len(cleaned)}.')
            model, ckpt_data, tokenizer, selected = load_model(checkpoint, device, config_path)
            result = score_variant(model, tokenizer, cleaned, pos, alt, device=selected)
            st.success('Score computed')
            c1, c2, c3, c4 = st.columns(4)
            llr = result['llr']
            delta = result['delta_loss']
            c1.metric('LLR (nats)', f'{llr:+.4f}' if llr is not None else 'N/A')
            c2.metric('Δloss (nats/base)', f'{delta:+.6f}')
            c3.metric('Ref loss', f'{result["loss_ref"]:.4f}')
            c4.metric('Mut loss', f'{result["loss_mut"]:.4f}')
            st.caption('LLR > 0 → model prefers reference (mutation is surprising). Δloss > 0 → mutation reduces overall predictability.')
            st.markdown('**Per-position probabilities at mutation site**')
            with st.spinner('Computing per-base probabilities...'):
                ids = tokenizer.encode(cleaned, unknown='n')[:pos + 1]
                x = torch.tensor([ids], device=selected)
                with torch.no_grad():
                    logits, _ = model(x)
                probs = torch.softmax(logits[0, -1], dim=-1)
                base_labels = tokenizer.dna_tokens
                prob_dict = {b: float(probs[tokenizer.stoi[b]]) for b in base_labels}
                pdf = pd.DataFrame([prob_dict])
                st.bar_chart(pdf, height=150)
        except Exception as exc:
            st.error(str(exc))


def render_evaluate_tab(checkpoint, device, fasta_path, config_path=None):
    st.subheader('Biological Quality Evaluation')
    st.caption('Compare real moss genome statistics against model-generated sequences.')

    fa_path = Path(fasta_path)
    if not fa_path.exists():
        st.warning(f'Real FASTA not found at {fa_path}. Download it first with `python scripts/fetch_genome.py`.')
        return

    col1, col2 = st.columns(2)
    with col1:
        n_real = st.number_input('Real windows', min_value=10, max_value=500, value=100, step=10, key='eval_n_real')
    with col2:
        n_gen = st.number_input('Generated sequences', min_value=10, max_value=500, value=100, step=10, key='eval_n_gen')

    if st.button('Run evaluation', type='primary', key='eval_run'):
        with st.spinner('Sampling real genome windows...'):
            real_seqs = sample_windows_from_fasta(fa_path, num_windows=n_real, window_size=256)

        with st.spinner('Generating sequences from model...'):
            model, ckpt_data, tokenizer, selected = load_model(checkpoint, device, config_path)
            gen_seqs = []
            for _ in range(n_gen):
                prefix_ids = tokenizer.encode('ACGT', unknown='n')
                idx = torch.tensor([prefix_ids], dtype=torch.long, device=selected)
                allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
                with torch.no_grad():
                    out = sample(model, idx, max_new_tokens=256, temperature=0.8, top_k=4, allowed_token_ids=allowed)
                gen_seqs.append(tokenizer.decode(out[0].tolist(), skip_special=True))

        with st.spinner('Computing metrics...'):
            result = evaluate_biological_quality(real_seqs, gen_seqs, model, tokenizer, selected)

        # GC distribution comparison
        st.markdown('### GC Content')
        real_gc = [gc_content(s) for s in real_seqs]
        gen_gc = [gc_content(s) for s in gen_seqs]
        gc_df = pd.DataFrame({'Real': real_gc, 'Generated': gen_gc})
        st.line_chart(gc_df, height=250)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Real GC mean', f'{result["gc"]["real"]["mean"]:.1%}')
        c2.metric('Gen GC mean', f'{result["gc"]["generated"]["mean"]:.1%}')
        c3.metric('Real GC std', f'{result["gc"]["real"]["std"]:.3f}')
        c4.metric('Gen GC std', f'{result["gc"]["generated"]["std"]:.3f}')

        st.markdown('### K-mer Divergence (Real vs Generated)')
        kmer_data = {
            'k=3': result['kmer_js_3'],
            'k=4': result['kmer_js_4'],
            'k=5': result['kmer_js_5'],
        }
        kmer_df = pd.DataFrame([kmer_data])
        st.bar_chart(kmer_df, height=200)
        st.caption('Lower JS divergence = more biologically realistic sequence')

        st.markdown('### Additional Metrics')
        c1, c2, c3 = st.columns(3)
        c1.metric('CpG O/E (Real)', f'{result["cpg"]["real"]:.4f}')
        c1.metric('CpG O/E (Gen)', f'{result["cpg"]["generated"]:.4f}')
        c2.metric('Entropy (Real)', f'{result["entropy"]["real"]:.4f}')
        c2.metric('Entropy (Gen)', f'{result["entropy"]["generated"]:.4f}')

        if 'uv_dimer' in result:
            st.markdown('### UV Dimer Analysis (Model Loss at Dimer Sites)')
            uv = result['uv_dimer']
            uv_df = pd.DataFrame([{
                'TT': uv['dimer_mean'].get('TT', 0),
                'TC': uv['dimer_mean'].get('TC', 0),
                'CT': uv['dimer_mean'].get('CT', 0),
                'CC': uv['dimer_mean'].get('CC', 0),
                'Other': uv['other_mean'],
            }])
            st.bar_chart(uv_df, height=200)
            st.caption('UV dimers (TT/TC/CT/CC) are pyrimidine dimer hotspots. Higher loss = model finds these sites more surprising.')


def render_graph_tab(checkpoint, device, fasta_path, config_path=None):
    st.subheader('Genome Graph Explorer')
    st.caption('Visualize how the model understands moss genome structure using graph theory.')

    fa_path = Path(fasta_path)
    if not fa_path.exists():
        st.warning(f'Real FASTA not found at {fa_path}. Download it first with `python scripts/fetch_genome.py`.')
        return

    viz_mode = st.radio('Visualization', ['Embedding Similarity Graph', 'UMAP Projection', 'De Bruijn Graph'], horizontal=True, label_visibility='collapsed')

    if viz_mode == 'Embedding Similarity Graph':
        n_windows = st.slider('Number of windows', min_value=20, max_value=300, value=100, step=10)
        threshold = st.slider('Similarity threshold', min_value=0.7, max_value=0.99, value=0.85, step=0.01, format='%.2f')
        if st.button('Build graph', type='primary', key='graph_build'):
            with st.spinner('Sampling genome windows...'):
                windows = sample_windows_from_fasta(fa_path, num_windows=n_windows, window_size=256)
            with st.spinner('Extracting embeddings...'):
                model, ckpt_data, tokenizer, selected = load_model(checkpoint, device, config_path)
                embs = extract_embeddings(model, windows, tokenizer, selected)
            with st.spinner('Building similarity graph...'):
                metadata = [{'gc': round(gc_content(s), 3)} for s in windows]
                G = build_similarity_graph(embs, windows, metadata=metadata, threshold=threshold)

            st.info(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
            try:
                from pyvis.network import Network
                net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#333333')
                net.from_nx(G)
                for node in G.nodes():
                    gc_val = G.nodes[node].get('gc', 0.5)
                    r = int(255 * (1 - gc_val))
                    b = int(255 * gc_val)
                    net.get_node(node)['color'] = f'rgb({r}, 100, {b})'
                    net.get_node(node)['size'] = max(5, gc_val * 30)
                with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                    net.save_graph(f.name)
                    with open(f.name) as fh:
                        html = fh.read()
                st.components.v1.html(html, height=620, scrolling=True)
                st.caption('Nodes colored by GC content: blue = low GC, red = high GC. Connected nodes have similar model embeddings.')
            except Exception as exc:
                st.error(f'PyVis error: {exc}')
                import matplotlib.pyplot as plt
                pos = __import__('networkx').spring_layout(G, seed=42)
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = [G.nodes[n].get('gc', 0.5) for n in G.nodes()]
                nx = __import__('networkx')
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, cmap='RdYlBu_r', node_size=50, alpha=0.8)
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)
                st.pyplot(fig)

    elif viz_mode == 'UMAP Projection':
        n_windows = st.slider('Number of windows', min_value=50, max_value=500, value=200, step=10)
        n_neighbors = st.slider('UMAP neighbors', min_value=5, max_value=50, value=15, step=5)
        if st.button('Compute UMAP', type='primary', key='umap_run'):
            with st.spinner('Sampling genome windows...'):
                windows = sample_windows_from_fasta(fa_path, num_windows=n_windows, window_size=256)
            with st.spinner('Extracting embeddings...'):
                model, ckpt_data, tokenizer, selected = load_model(checkpoint, device, config_path)
                embs = extract_embeddings(model, windows, tokenizer, selected)
            with st.spinner('Computing UMAP projection...'):
                embedding_2d = compute_umap(embs, n_neighbors=n_neighbors)

            gcs = [gc_content(s) for s in windows]
            umap_df = pd.DataFrame({
                'UMAP1': embedding_2d[:, 0],
                'UMAP2': embedding_2d[:, 1],
                'GC content': gcs,
                'Length': [len(s) for s in windows],
            })
            st.scatter_chart(umap_df, x='UMAP1', y='UMAP2', color='GC content', size='Length', height=600)
            st.caption('Each point = a genome window, embedded by the model into 2D space. Color = GC content.')

    elif viz_mode == 'De Bruijn Graph':
        st.markdown('### De Bruijn Graph: Real vs Generated DNA')
        k_size = st.selectbox('k-mer size', [3, 4, 5, 6], index=1)
        sample_seq = st.text_area('DNA sequence (or leave blank to auto-generate)', value='', height=80, help='Enter a DNA sequence or leave blank to generate one from the model.')
        if st.button('Generate & Compare', type='primary', key='db_run'):
            model, ckpt_data, tokenizer, selected = load_model(checkpoint, device, config_path)
            if sample_seq.strip():
                real_seq = clean_dna(sample_seq)
            else:
                with st.spinner('Generating sequence from model...'):
                    prefix_ids = tokenizer.encode('ACGTACGT', unknown='n')
                    idx = torch.tensor([prefix_ids], dtype=torch.long, device=selected)
                    allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
                    with torch.no_grad():
                        out = sample(model, idx, max_new_tokens=100, temperature=0.8, top_k=4, allowed_token_ids=allowed)
                    real_seq = tokenizer.decode(out[0].tolist(), skip_special=True)

            with st.spinner('Building De Bruijn graphs...'):
                G_real = build_debruijn_graph(real_seq, k=k_size)
                prefix_ids = tokenizer.encode(real_seq[:min(8, len(real_seq))], unknown='n')
                idx = torch.tensor([prefix_ids], dtype=torch.long, device=selected)
                allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
                with torch.no_grad():
                    out = sample(model, idx, max_new_tokens=len(real_seq) - len(prefix_ids), temperature=0.8, top_k=4, allowed_token_ids=allowed)
                gen_seq = tokenizer.decode(out[0].tolist(), skip_special=True)
                G_gen = build_debruijn_graph(gen_seq, k=k_size)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'**Real sequence** ({len(real_seq)} bp)')
                st.markdown(f'Nodes: {G_real.number_of_nodes()}, Edges: {G_real.number_of_edges()}')

            with col2:
                st.markdown(f'**Generated sequence** ({len(gen_seq)} bp)')
                st.markdown(f'Nodes: {G_gen.number_of_nodes()}, Edges: {G_gen.number_of_edges()}')

            with st.spinner('Rendering graphs...'):
                from pyvis.network import Network
                for label, G, seq in [('Real', G_real, real_seq), ('Generated', G_gen, gen_seq)]:
                    st.markdown(f'**{label} De Bruijn graph (k={k_size})**')
                    net = Network(height='400px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)
                    net.from_nx(G)
                    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                        net.save_graph(f.name)
                        with open(f.name) as fh:
                            html = fh.read()
                    st.components.v1.html(html, height=420, scrolling=True)
                st.caption(f'Real: {real_seq[:80]}...  |  Generated: {gen_seq[:80]}...')


def main() -> None:
    cli = parse_cli_args()
    st.set_page_config(page_title='moss-dna-gpt', page_icon='🧬', layout='wide')
    st.title('moss-dna-gpt')
    st.caption('DNA language model toolkit for Physcomitrium patens')

    tab_names = ['🧬 Generate', '📊 Score', '📈 Evaluate', '🔗 Graph']
    tab_gen, tab_score, tab_eval, tab_graph = st.tabs(tab_names)

    with st.sidebar:
        st.header('Checkpoint')
        checkpoint = st.text_input('Checkpoint path', value=cli.checkpoint, key='checkpoint_path_input')
        config_path = cli.config
        if checkpoint.endswith('.safetensors'):
            config_path = st.text_input('Config path', value=cli.config or str(Path(checkpoint).parent / 'config.json'), key='config_path_input')
        device_options = ['auto', 'cpu', 'cuda', 'mps']
        default_device_index = device_options.index(cli.device) if cli.device in device_options else 0
        device = st.selectbox('Device', device_options, index=default_device_index)
        st.header('Sampling')
        temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=0.8, step=0.05)
        top_k_value = st.number_input('Top-k, 0 = disabled', min_value=0, max_value=9, value=0, step=1)
        top_k = int(top_k_value) if top_k_value else None
        sliding = st.checkbox('Sliding window mode (long generation)', key='sliding_mode')
        total_tokens = 10240
        chunk_size = 1024
        max_new_tokens = 256
        if sliding:
            total_tokens = st.number_input('Total tokens to generate', min_value=1024, max_value=100000, value=10240, step=1024, key='total_tokens')
            chunk_size = st.selectbox('Chunk size', [512, 1024, 2048, 4096], index=1, key='chunk_size')
        else:
            max_new_tokens = st.slider('Max new bases', min_value=1, max_value=2048, value=256, step=1, key='max_new_tokens')
        st.header('Safety')
        st.caption('The model only predicts next DNA bases. Do not interpret output as SNP effect, gene function, or adaptation prediction.')

    render_learning_curve()
    config_path = st.session_state.get('config_path_input', cli.config)

    with tab_gen:
        render_generate_tab(checkpoint, device, temperature, top_k, sliding, total_tokens, chunk_size, max_new_tokens, cli, config_path)
    with tab_score:
        render_variant_tab(checkpoint, device, config_path)
    with tab_eval:
        render_evaluate_tab(checkpoint, device, cli.fasta, config_path)
    with tab_graph:
        render_graph_tab(checkpoint, device, cli.fasta, config_path)


if __name__ == '__main__':
    main()
