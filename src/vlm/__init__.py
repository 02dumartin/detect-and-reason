__all__ = ["run_vlm_classification"]


def __getattr__(name: str):
    if name == "run_vlm_classification":
        from .classification import run_vlm_classification

        return run_vlm_classification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
