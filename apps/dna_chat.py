from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys

import torch
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.sampling import sample
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.trainer import load_checkpoint, resolve_device

DNA_RE = re.compile(r'[^ACGTNacgtn]+')
DEFAULT_CHECKPOINT = 'runs/physcomitrium_patens_quick_256/ckpt_step_50.pt'


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


def main() -> None:
    cli = parse_cli_args()
    st.set_page_config(page_title='moss-dna-gpt', page_icon='🧬', layout='wide')
    st.title('moss-dna-gpt DNA completion UI')
    st.caption('This is a DNA prefix-completion interface. It is not a natural-language biology assistant.')

    with st.sidebar:
        st.header('Checkpoint')
        checkpoint = st.text_input('Checkpoint path', value=cli.checkpoint)
        device_options = ['auto', 'cpu', 'cuda']
        default_device_index = device_options.index(cli.device) if cli.device in device_options else 0
        device = st.selectbox('Device', device_options, index=default_device_index)
        st.header('Sampling')
        max_new_tokens = st.slider('Max new bases', min_value=1, max_value=2048, value=256, step=1)
        temperature = st.slider('Temperature', min_value=0.1, max_value=2.0, value=0.8, step=0.05)
        top_k_value = st.number_input('Top-k, 0 = disabled', min_value=0, max_value=9, value=0, step=1)
        top_k = int(top_k_value) if top_k_value else None
        st.header('Safety boundary')
        st.write('The model only predicts next DNA bases. Do not interpret output as SNP effect, gene function, or adaptation prediction.')

    if 'history' not in st.session_state:
        st.session_state.history = []

    default_prefix = 'ACGTACGTACGT'
    prefix = st.text_area('DNA prefix', value=default_prefix, height=120, help='Only A/C/G/T/N are kept. Other characters are removed.')

    col_a, col_b = st.columns([1, 1])
    with col_a:
        run = st.button('Generate continuation', type='primary')
    with col_b:
        clear = st.button('Clear history')
    if clear:
        st.session_state.history = []

    if run:
        try:
            full, stats, selected_device, step, param_count = generate_continuation(
                checkpoint,
                prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )
            item = {
                'prefix': clean_dna(prefix),
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

    for i, item in enumerate(st.session_state.history, start=1):
        with st.container(border=True):
            st.subheader(f'Generation {i}')
            st.write(f"device: `{item['device']}`  step: `{item['step']}`  params: `{item['param_count']}`  temperature: `{item['temperature']}`  top_k: `{item['top_k']}`")
            st.text_area('Output DNA', value=item['output'], height=160, key=f'out_{i}')
            st.json(item['stats'])


if __name__ == '__main__':
    main()
