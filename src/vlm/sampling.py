"""시드 고정 랜덤 샘플링 유틸리티.

VLM 데이터셋/exemplar 준비 과정에서 무작위성이 필요한 지점(현재는 클래스
다운샘플링)을 한 곳에 모아, 항상 명시적인 seed로만 재현 가능하게 한다.
"""

from __future__ import annotations

import random
from typing import Any


def downsample_class_records(
    records: list[dict[str, Any]],
    *,
    class_name: str,
    keep_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    """특정 클래스의 record만 keep_ratio 비율로 무작위 다운샘플링한다.

    같은 seed로 호출하면 항상 동일한 부분집합이 남는다.
    """
    if keep_ratio >= 1.0:
        return list(records)
    if keep_ratio <= 0.0:
        raise ValueError("keep_ratio must be greater than 0.0")

    matching_indices = [idx for idx, record in enumerate(records) if record["class_name"] == class_name]
    if not matching_indices:
        return list(records)

    keep_count = max(1, int(round(len(matching_indices) * keep_ratio)))
    rng = random.Random(seed)
    shuffled = matching_indices[:]
    rng.shuffle(shuffled)
    keep_indices = set(shuffled[:keep_count])
    return [
        record
        for idx, record in enumerate(records)
        if record["class_name"] != class_name or idx in keep_indices
    ]
