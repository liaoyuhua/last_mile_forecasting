"""Explicit backbones: the demo never impersonates TimesFM."""

from threading import Lock
import math
import os
from pathlib import Path
from .data import day

_model = None
_model_lock = Lock()


def seasonal_baseline(history, target):
    recent = history[-56:]
    means = {
        weekday: sum(r["y"] for r in recent if day(r["ds"]).weekday() == weekday)
        / sum(day(r["ds"]).weekday() == weekday for r in recent)
        for weekday in range(7)
    }
    return [round(means[day(ds).weekday()], 4) for ds in target]


def forecast(history, target, backbone, context_length=1024):
    if backbone == "seasonal":
        return seasonal_baseline(history, target)
    if backbone != "timesfm":
        raise ValueError("Unknown backbone")
    try:
        import numpy as np
        import timesfm
    except ImportError:
        raise ValueError(
            "TimesFM is not installed. Run pip install -e '.[timesfm]'."
        ) from None
    global _model
    with _model_lock:
        if _model is None:
            cache = os.environ.get(
                "LAST_MILE_MODEL_CACHE",
                str(Path(__file__).resolve().parent.parent / "runtime/models"),
            )
            _model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch", cache_dir=cache
            )
        _model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=math.ceil(len(target) / 128) * 128,
                normalize_inputs=True,
                per_core_batch_size=1,
                infer_is_positive=True,
            )
        )
        point, _ = _model.forecast(
            horizon=len(target),
            inputs=[
                np.array([r["y"] for r in history[-context_length:]], dtype=np.float32)
            ],
        )
        return [float(x) for x in point[0][: len(target)]]
