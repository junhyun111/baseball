"""Retained V2.1 Pitcher Expert experiment.

V2.2 deliberately does not import or package this source.  The executable
historical implementation remains in :mod:`train_v21`; this module exposes
its feature definition from an explicit experimental namespace so Trackman
features can be attached later without changing the V2.2 production path.
"""

from __future__ import annotations

from train_v21 import make_feature_sets_v21


def pitcher_feature_spec(
    columns: list[str], categorical_columns: list[str]
) -> tuple[list[str], list[str]]:
    feature_sets, categorical_sets = make_feature_sets_v21(
        columns, categorical_columns
    )
    return feature_sets["pitcher"], categorical_sets["pitcher"]
