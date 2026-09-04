from .survival_flow import run_survival_flow

REGISTRY = {
    "survival_flow": {
        "function": run_survival_flow,
        "supported_pipelines": ["internal_cv"],
    },
}

__all__ = ["REGISTRY"]