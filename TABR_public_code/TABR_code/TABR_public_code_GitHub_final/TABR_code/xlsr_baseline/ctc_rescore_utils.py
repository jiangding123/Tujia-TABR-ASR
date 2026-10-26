#!/usr/bin/env python3
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


NEG_INF = -float("inf")


def log_add(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = a if a >= b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_size: int,
    blank_id: int = 0,
    allowed_token_ids: Sequence[int] = None,
    token_prune: int = 20,
) -> List[Tuple[Tuple[int, ...], float]]:
    """
    Standard CTC prefix beam search.

    log_probs:
        [T, V] log-softmax probabilities.

    Returns:
        [(collapsed_label_prefix, prefix_log_score), ...], best first.
        Returned prefixes contain no blank symbols.

    token_prune:
        Per-frame top-k over valid output labels + blank. 0 disables pruning.
        This is only a candidate-generation speed optimization. After N-best
        generation, every retained candidate is re-scored with exact CTC
        sequence probability by ctc_sequence_log_scores().
    """
    if log_probs.ndim != 2:
        raise ValueError(
            f"log_probs must be [T,V], got {tuple(log_probs.shape)}"
        )
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")

    _, vocab_size = log_probs.shape

    if allowed_token_ids is None:
        allowed = list(range(vocab_size))
    else:
        allowed = sorted(set(int(x) for x in allowed_token_ids))

    if blank_id not in allowed:
        allowed.append(blank_id)
    allowed = sorted(set(allowed))

    allowed_t = torch.tensor(
        allowed, dtype=torch.long, device=log_probs.device
    )

    # prefix -> (probability ending in blank,
    #            probability ending in nonblank)
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {
        (): (0.0, NEG_INF)
    }

    for t in range(log_probs.size(0)):
        frame_allowed = log_probs[t].index_select(0, allowed_t)

        if token_prune and token_prune < len(allowed):
            values, local_ids = torch.topk(
                frame_allowed, k=token_prune
            )
            frame_ids = allowed_t.index_select(
                0, local_ids
            ).tolist()
            frame_values = values.tolist()

            # Blank must always be considered even if it was not in top-k.
            if blank_id not in frame_ids:
                frame_ids.append(blank_id)
                frame_values.append(
                    float(log_probs[t, blank_id].item())
                )
        else:
            frame_ids = allowed
            frame_values = frame_allowed.tolist()

        token_scores = list(zip(frame_ids, frame_values))
        next_beams: Dict[
            Tuple[int, ...], Tuple[float, float]
        ] = {}

        def get(prefix):
            return next_beams.get(
                prefix, (NEG_INF, NEG_INF)
            )

        def add_blank(prefix, value):
            old_pb, old_pnb = get(prefix)
            next_beams[prefix] = (
                log_add(old_pb, value),
                old_pnb,
            )

        def add_nonblank(prefix, value):
            old_pb, old_pnb = get(prefix)
            next_beams[prefix] = (
                old_pb,
                log_add(old_pnb, value),
            )

        for prefix, (pb, pnb) in beams.items():
            total = log_add(pb, pnb)

            for token, lp in token_scores:
                token = int(token)
                lp = float(lp)

                if token == blank_id:
                    add_blank(prefix, total + lp)
                    continue

                if prefix and token == prefix[-1]:
                    # Same label without an intervening blank collapses to
                    # the existing output label.
                    if pnb != NEG_INF:
                        add_nonblank(
                            prefix, pnb + lp
                        )

                    # Same label after a blank forms a repeated output label,
                    # e.g. "a a".
                    if pb != NEG_INF:
                        add_nonblank(
                            prefix + (token,),
                            pb + lp,
                        )
                else:
                    if total != NEG_INF:
                        add_nonblank(
                            prefix + (token,),
                            total + lp,
                        )

        ranked = sorted(
            next_beams.items(),
            key=lambda kv: log_add(
                kv[1][0], kv[1][1]
            ),
            reverse=True,
        )[:beam_size]

        beams = dict(ranked)

    result = [
        (
            prefix,
            log_add(pb, pnb),
        )
        for prefix, (pb, pnb) in beams.items()
    ]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def ctc_sequence_log_scores(
    log_probs: torch.Tensor,
    targets: Sequence[Sequence[int]],
    blank_id: int = 0,
) -> List[float]:
    """
    Exact CTC log P(target | audio) for multiple fixed target sequences.

    log_probs:
        [T, C] log-softmax probabilities.

    targets:
        List of label-ID sequences. Targets must not contain blank IDs.

    Returns:
        One raw sequence log probability per target.
    """
    if log_probs.ndim != 2:
        raise ValueError(
            f"log_probs must be [T,C], got {tuple(log_probs.shape)}"
        )
    if not targets:
        return []

    device = log_probs.device
    time_steps, classes = log_probs.shape
    num_targets = len(targets)

    target_lengths_list = [
        len(seq) for seq in targets
    ]
    max_target_len = max(
        max(target_lengths_list), 1
    )

    target_tensor = torch.zeros(
        (num_targets, max_target_len),
        dtype=torch.long,
        device=device,
    )

    for i, seq in enumerate(targets):
        if seq:
            target_tensor[
                i, : len(seq)
            ] = torch.tensor(
                list(seq),
                dtype=torch.long,
                device=device,
            )

    input_lengths = torch.full(
        (num_targets,),
        time_steps,
        dtype=torch.long,
        device=device,
    )
    target_lengths = torch.tensor(
        target_lengths_list,
        dtype=torch.long,
        device=device,
    )

    batched_log_probs = (
        log_probs
        .unsqueeze(1)
        .expand(
            time_steps,
            num_targets,
            classes,
        )
        .contiguous()
    )

    # Match the safe/native CTC path already used by the training script.
    with torch.backends.cudnn.flags(enabled=False):
        losses = F.ctc_loss(
            batched_log_probs,
            target_tensor,
            input_lengths,
            target_lengths,
            blank=blank_id,
            reduction="none",
            zero_infinity=False,
        )

    return [
        float(x)
        for x in (-losses).detach().cpu().tolist()
    ]
