from __future__ import annotations

import torch
import torch.nn.functional as F

from .tokenizer import DnaTokenizer
from .metrics import nats_to_bits


def score_variant(
    model: torch.nn.Module,
    tokenizer: DnaTokenizer,
    seq: str,
    pos: int,
    alt: str,
    device: str = "cpu",
) -> dict:
    if not (0 <= pos < len(seq)):
        raise ValueError(f"pos={pos} out of range for sequence of length {len(seq)}")
    if alt not in tokenizer.stoi:
        raise ValueError(f"alt={alt!r} is not a valid token")

    model.eval()
    ids = tokenizer.encode(seq, unknown="n")
    seq_len = len(ids)

    # Per-position LLR (requires left context)
    llr: float | None = None
    if pos > 0:
        ctx = torch.tensor([ids[:pos]], device=device)
        with torch.no_grad():
            logits, _ = model(ctx)
        log_probs = F.log_softmax(logits[0, -1], dim=-1)
        ref_id = ids[pos]
        alt_id = tokenizer.stoi[alt]
        log_p_ref = log_probs[ref_id].item()
        log_p_alt = log_probs[alt_id].item()
        llr = log_p_ref - log_p_alt

    # Full-sequence Δloss
    x_ref = torch.tensor([ids[:-1]], device=device)
    y_ref = torch.tensor([ids[1:]], device=device)
    with torch.no_grad():
        _, loss_ref_t = model(x_ref, y_ref)
    loss_ref = loss_ref_t.item()

    seq_mut = seq[:pos] + alt + seq[pos + 1:]
    mut_ids = tokenizer.encode(seq_mut, unknown="n")
    x_mut = torch.tensor([mut_ids[:-1]], device=device)
    y_mut = torch.tensor([mut_ids[1:]], device=device)
    with torch.no_grad():
        _, loss_mut_t = model(x_mut, y_mut)
    loss_mut = loss_mut_t.item()

    delta_loss = loss_mut - loss_ref

    return {
        "llr": llr,
        "llr_bits": nats_to_bits(llr) if llr is not None else None,
        "loss_ref": loss_ref,
        "loss_ref_bits": nats_to_bits(loss_ref),
        "loss_mut": loss_mut,
        "loss_mut_bits": nats_to_bits(loss_mut),
        "delta_loss": delta_loss,
        "delta_loss_bits": nats_to_bits(delta_loss),
        "ref_base": seq[pos],
        "alt_base": alt,
        "position": pos,
        "seq_length": seq_len,
    }
