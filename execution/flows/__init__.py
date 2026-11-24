from collections.abc import Callable
from typing import Any, TypedDict

from .survival_flow import run_survival_flow


class FlowSpec(TypedDict):
    function: Callable[..., Any]
    supported_pipelines: list[str]


REGISTRY: dict[str, FlowSpec] = {
    "survival_flow": {
        "function": run_survival_flow,
        "supported_pipelines": ["internal_cv"],
    },
}

__all__ = ["REGISTRY"]
