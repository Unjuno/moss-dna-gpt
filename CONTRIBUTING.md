# Contributing

## Development setup

```bash
git clone https://github.com/Unjuno/moss-dna-gpt.git
cd moss-dna-gpt
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Code style

- Keep code readable and concise.
- Avoid semicolons for multi-statement lines.
- Add docstrings to public functions (Google style preferred).
- Use type hints (`from __future__ import annotations`).

## Pull request process

1. Create a feature branch from `main`.
2. Make changes and ensure `pytest` passes.
3. Update or add tests as needed.
4. Submit a PR with a clear description.

## Publishing to HuggingFace

```bash
# Convert checkpoint to safetensors
python scripts/convert_to_safetensors.py --checkpoint path/to/ckpt.pt --out-dir repo/moss-dna-gpt-20m-patens

# Upload to HF Hub
HF_TOKEN=your_token python scripts/publish_to_hf.py --repo-id your-org/moss-dna-gpt-20m
```
