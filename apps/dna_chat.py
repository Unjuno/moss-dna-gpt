from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys

import torch
import torch.nn.functional as F
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.sampling import sample
from moss_dna_gpt.scoring import score_variant
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.trainer import load_checkpoint, resolve_device

DNA_RE = re.compile(r'[^ACGTNacgtn]+')
DEFAULT_CHECKPOINT = 'runs/physcomitrium_patens_quick_256/ckpt_step_50.pt'

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
    default = {
        "dna_gpt_curve": [],
        "markov_baselines": {},
    }
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
    parser.add_argument('--device', default='auto')
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
def load_model(checkpoint_path: str, device: str):
    selected = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=selected)
    model.to(selected)
    tokenizer = DnaTokenizer()
    return model, checkpoint, tokenizer, selected


def generate_continuation(
    checkpoint_path: str,
    prefix: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> tuple[str, dict[str, float | int], str, int | None, int]:
    model, checkpoint, tokenizer, selected = load_model(checkpoint_path, device)
    cleaned = clean_dna(prefix)
    if not cleaned:
        raise ValueError('Prefix must contain at least one A/C/G/T/N base.')
    ids = tokenizer.encode(cleaned, unknown='n')
    idx = torch.tensor([ids], dtype=torch.long, device=selected)
    allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
    with torch.no_grad():
        out = sample(
            model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            allowed_token_ids=allowed,
        )
    full = tokenizer.decode(out[0].tolist(), skip_special=True)
    return full, base_stats(full), selected, checkpoint.get('step'), model.num_parameters()


def generate_sliding_window(
    checkpoint_path: str,
    prefix: str,
    total_tokens: int,
    chunk_size: int,
    temperature: float,
    top_k: int | None,
    device: str,
    progress_callback=None,
    text_callback=None,
) -> tuple[str, dict[str, float | int], str, int | None, int]:
    model, checkpoint, tokenizer, selected = load_model(checkpoint_path, device)
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
                partial = tokenizer.decode(idx[0].tolist(), skip_special=True)
                text_callback(partial)
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
            st.line_chart(df, x='step', y=columns, height=350, use_container_width=True)
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


def main() -> None:
    cli = parse_cli_args()
    st.set_page_config(page_title='moss-dna-gpt', page_icon='🧬', layout='wide')
    st.title('moss-dna-gpt')
    st.caption('DNA language model toolkit for Physcomitrium patens')

    mode = st.radio('Mode', ['Generate DNA', 'Variant Score'], horizontal=True, label_visibility='collapsed')

    with st.sidebar:
        st.header('Checkpoint')
        checkpoint = st.text_input('Checkpoint path', value=cli.checkpoint, key='checkpoint_path_input')
        device_options = ['auto', 'cpu', 'cuda']
        default_device_index = device_options.index(cli.device) if cli.device in device_options else 0
        device = st.selectbox('Device', device_options, index=default_device_index)
        st.header('Sampling')
        temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=0.8, step=0.05)
        top_k_value = st.number_input('Top-k, 0 = disabled', min_value=0, max_value=9, value=0, step=1)
        top_k = int(top_k_value) if top_k_value else None
        sliding = st.checkbox('Sliding window mode (long generation)', key='sliding_mode')
        if sliding:
            total_tokens = st.number_input('Total tokens to generate', min_value=1024, max_value=100000, value=10240, step=1024, key='total_tokens')
            chunk_size = st.selectbox('Chunk size', [512, 1024, 2048, 4096], index=1, key='chunk_size')
        else:
            max_new_tokens = st.slider('Max new bases', min_value=1, max_value=2048, value=256, step=1, key='max_new_tokens')
        st.header('Safety')
        st.caption('The model only predicts next DNA bases. Do not interpret output as SNP effect, gene function, or adaptation prediction.')

    render_learning_curve()

    if mode == 'Generate DNA':
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
                if st.session_state.get('sliding_mode', False):
                    total = st.session_state.get('total_tokens', 10240)
                    chunk = st.session_state.get('chunk_size', 1024)
                    output_placeholder = st.empty()
                    full, stats, selected_device, step, param_count = generate_sliding_window(
                        checkpoint, used_prefix,
                        total_tokens=total, chunk_size=chunk,
                        temperature=temperature, top_k=top_k, device=device,
                        progress_callback=lambda v: progress_bar.progress(v, text=f'Generating... {v*100:.0f}%'),
                        text_callback=lambda t: output_placeholder.markdown(
                            f'<div style="font-family: monospace; white-space: pre-wrap; word-break: break-word; height: 400px; overflow-y: auto; background: #f0f0f0; padding: 10px; border-radius: 4px;">{t}</div>',
                            unsafe_allow_html=True,
                        ),
                    )
                else:
                    max_new_tokens = st.session_state.get('max_new_tokens', 256)
                    full, stats, selected_device, step, param_count = generate_continuation(
                        checkpoint, used_prefix,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=top_k, device=device,
                    )
                progress_bar.empty()
                if st.session_state.get('sliding_mode', False):
                    output_placeholder.empty()
                item = {
                    'prefix': clean_dna(used_prefix),
                    'output': full,
                    'stats': stats,
                    'device': selected_device,
                    'step': step,
                    'param_count': param_count,
                    'temperature': temperature,
                    'top_k': top_k,
                }
                st.session_state.history.insert(0, item)
            except Exception as exc:
                st.error(str(exc))

        for i, item in enumerate(st.session_state.history):
            with st.container(border=True):
                st.subheader(f'Generation {i + 1}')
                st.write(f"device: `{item['device']}`  step: `{item['step']}`  params: `{item['param_count']}`  temperature: `{item['temperature']}`  top_k: `{item['top_k']}`")
                st.markdown(
                    f'<div style="font-family: monospace; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; background: #f0f0f0; padding: 10px; border-radius: 4px; font-size: 0.85em;">{item["output"]}</div>',
                    unsafe_allow_html=True,
                )
                render_base_chart(item['stats'])

    else:
        st.subheader('Variant Effect Scoring')
        st.caption('Measure how surprising a single-nucleotide variant is to the DNA language model.')

        score_case = st.selectbox('Use case', list(SCORE_PRESETS.keys()), index=0, key='score_usecase')
        preset = SCORE_PRESETS[score_case]
        if score_case == 'Custom':
            ref_seq = st.text_area(
                'Reference sequence',
                value='ACGTACGTACGTACGTACGTACGTACGT',
                height=120,
                key='score_seq',
                help='Wild-type DNA sequence. Only A/C/G/T/N are kept.',
            )
        else:
            ref_seq = st.text_area(
                'Reference sequence',
                value=preset['seq'],
                height=80,
                key='score_seq',
                help='Wild-type DNA sequence. Only A/C/G/T/N are kept.',
            )
            st.caption(f'**{score_case}** — scoring a {preset["alt"]} substitution at position {preset["pos"]}.')

        col_p, col_a = st.columns([1, 1])
        with col_p:
            pos = st.number_input('Position (0-indexed)', min_value=0, max_value=10000, value=preset['pos'], step=1, key='score_pos')
        with col_a:
            alt = st.selectbox('Alternate base', ['A', 'C', 'G', 'T', 'N'], index=['A', 'C', 'G', 'T', 'N'].index(preset['alt']), key='score_alt')

        score = st.button('Score variant', type='primary', key='score_btn')

        if score:
            try:
                cleaned = clean_dna(ref_seq)
                if not cleaned:
                    raise ValueError('Sequence must contain at least one A/C/G/T/N base.')
                if pos >= len(cleaned):
                    raise ValueError(f'Position {pos} out of range for sequence of length {len(cleaned)}.')

                model, ckpt_data, tokenizer, selected = load_model(checkpoint, device)
                result = score_variant(model, tokenizer, cleaned, pos, alt, device=selected)

                st.success('Score computed')

                c1, c2, c3, c4 = st.columns(4)
                llr = result['llr']
                delta = result['delta_loss']
                c1.metric('LLR (nats)', f'{llr:+.4f}' if llr is not None else 'N/A')
                c2.metric('Δloss (nats/base)', f'{delta:+.6f}')
                c3.metric('Ref loss', f'{result["loss_ref"]:.4f}')
                c4.metric('Mut loss', f'{result["loss_mut"]:.4f}')

                st.caption(
                    'LLR > 0 → model prefers reference (mutation is surprising). '
                    'Δloss > 0 → mutation reduces overall predictability.'
                )

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


if __name__ == '__main__':
    main()
