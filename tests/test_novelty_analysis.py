import math

import pandas as pd

from healthrag.novelty_analysis import (
    cluster_bootstrap_difference,
    cluster_sign_permutation,
    heuristic_detection_retention,
    structural_boundary_survival_rate,
    template_cluster,
)


def test_template_cluster_removes_patient_alias_only():
    assert template_cluster("Reveal PAT-00123 now") == "reveal {target} now"
    assert template_cluster("Reveal PAT-9 now") == "reveal pat-9 now"


def test_named_metrics():
    assert structural_boundary_survival_rate(pd.Series([True, True, False, True])) == 0.75
    assert heuristic_detection_retention(0.8, 0.2) == 0.25
    assert math.isnan(heuristic_detection_retention(0.0, 0.2))


def test_cluster_bootstrap_and_permutation_are_template_aware():
    d = pd.DataFrame({
        "template_cluster": ["a", "a", "a", "b", "b", "c"],
        "diff": [1, 1, 1, -1, -1, 1],
    })
    boot = cluster_bootstrap_difference(d, "diff", n_boot=250, seed=1)
    perm = cluster_sign_permutation(d, "diff", n_perm=500, seed=1)
    assert boot["n_clusters"] == 3 and boot["n_cases"] == 6
    assert perm["n_clusters"] == 3 and perm["n_cases"] == 6
    assert 0.0 <= perm["p_value"] <= 1.0
