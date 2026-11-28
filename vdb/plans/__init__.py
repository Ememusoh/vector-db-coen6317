from .base import Plan, PlanResult
from .context import AttributeIndex, FilterContext, MultiLabelContext
from .insearch import HNSWInSearch, IVFInSearch, PartitionedIVFInSearch
from .postfilter import HNSWPostFilter, IVFPostFilter
from .prefilter import PreFilter

ALL_PLANS = (PreFilter, IVFPostFilter, IVFInSearch, PartitionedIVFInSearch, HNSWPostFilter, HNSWInSearch)


def make_plans(ctx, names=None):
    """Instantiate every plan the context supports (or only `names`)."""
    plans = []
    for cls in ALL_PLANS:
        if names and cls.name not in names:
            continue
        if cls.index in ("ivf",) and ctx.ivf is None:
            continue
        if cls.index == "ivf_part" and ctx.pivf is None:
            continue
        if cls.index == "hnsw" and ctx.hnsw is None:
            continue
        plans.append(cls(ctx))
    return plans


__all__ = ["Plan", "PlanResult", "AttributeIndex", "FilterContext", "MultiLabelContext", "PreFilter", "IVFPostFilter",
           "HNSWPostFilter", "IVFInSearch", "PartitionedIVFInSearch", "HNSWInSearch", "ALL_PLANS", "make_plans"]
