#!/usr/bin/env python3
import re
from typing import Dict, Tuple

from common import SPECIAL_TOKENS


def split_full_token(
    token: str,
    tone_regex: str,
    unmarked_tone_token: str,
) -> Tuple[str, str]:
    """
    Examples:
        long3 -> ("long", "3")
        a     -> ("a", "<no_tone>")
    """
    pattern = re.compile(tone_regex)
    m = pattern.fullmatch(token)

    if m is None:
        return token, unmarked_tone_token

    base = m.group(1)
    tone = m.group(2)

    if not base or not tone:
        raise ValueError(
            f"invalid full token split: {token!r}"
        )

    return base, tone


def build_main_to_aux_projection(
    id2token: Dict[int, str],
    notone_token2id: Dict[str, int],
    tone_token2id: Dict[str, int],
    tone_regex: str,
    unmarked_tone_token: str,
):
    """
    Reconstruct exactly the same mapping rule used during XLS-R
    auxiliary-CTC training.

    Important:
    The numeric aux_notone/aux_tone IDs stored in aligned data.list files
    belong to another auxiliary label space. Rescoring must use the
    token2id maps saved inside auxiliary_heads.pt.
    """
    main_to_notone = {}
    main_to_tone = {}
    split_map = {}

    for idx in sorted(id2token):
        token = id2token[idx]

        if token in SPECIAL_TOKENS:
            continue

        base, tone = split_full_token(
            token,
            tone_regex=tone_regex,
            unmarked_tone_token=unmarked_tone_token,
        )

        if base not in notone_token2id:
            raise KeyError(
                f"no-tone token {base!r}, derived from {token!r}, "
                "is missing from auxiliary_heads.pt"
            )

        if tone not in tone_token2id:
            raise KeyError(
                f"tone token {tone!r}, derived from {token!r}, "
                "is missing from auxiliary_heads.pt"
            )

        main_to_notone[idx] = int(
            notone_token2id[base]
        )
        main_to_tone[idx] = int(
            tone_token2id[tone]
        )
        split_map[token] = (base, tone)

    return (
        main_to_notone,
        main_to_tone,
        split_map,
    )
