from healthrag.core import AuthorizationMatrix, FuzzyRiskController, PatientRecord, extract_target_aliases
from healthrag.prompts import (
    generate_cases,
    generate_heldout_cases,
    generate_regression_cases,
    generate_resolution_cases,
    session_telemetry,
    session_trust,
)


def _fixture(n=1000):
    width = max(2, len(str(n)))
    records = [
        PatientRecord(
            f"PAT-{i:0{width}d}",
            str(i),
            f"PatientAlias: PAT-{i:0{width}d}\nSyntheticName: Person {i}\nPrimaryCondition: cond{i}\nPrimaryMedication: med{i}",
            f"C{i}",
            f"cond{i}",
            f"med{i}",
        )
        for i in range(1, n + 1)
    ]
    aliases = [r.alias for r in records]
    acl = AuthorizationMatrix({
        "A": aliases[0:200], "B": aliases[200:400], "C": aliases[400:600], "D": aliases[600:800], "E": [],
    })
    return records, acl


def test_publication_benchmark_shape():
    records, acl = _fixture()
    cases = generate_cases(records, acl)
    assert len(cases) == 3000
    assert sum(c.is_legitimate for c in cases) == 1000
    assert sum(c.category == "direct_unauthorized" for c in cases) == 1000
    assert sum(c.category == "authorized_suspicious" for c in cases) == 250
    assert sum(not c.is_authorized for c in cases) == 1750
    assert len({c.case_id for c in cases}) == 3000


def test_session_trust_is_label_independent_and_deterministic():
    # Category is deliberately not an argument: identical principal/target pairs
    # have identical simulated telemetry regardless of which benchmark uses them.
    a = session_telemetry("A", "PAT-0001")
    b = session_telemetry("A", "PAT-0001")
    assert a == b
    assert session_trust("A", "PAT-0001") == a["trust"]
    assert 0.25 <= a["trust"] <= 0.95


def test_heldout_benchmark_is_vocabulary_disjoint_from_frozen_detector():
    records, acl = _fixture()
    cases = generate_heldout_cases(records, acl)
    assert len(cases) == 300
    assert sum(c.is_authorized for c in cases) == 150
    assert sum(not c.is_authorized for c in cases) == 150
    assert all(FuzzyRiskController.prompt_injection_risk(c.prompt) == 0.0 for c in cases)


def test_alias_free_resolution_suite_has_no_parseable_aliases():
    records, acl = _fixture()
    cases = generate_resolution_cases(records, acl)
    assert len(cases) == 200
    assert sum(c.is_legitimate for c in cases) == 100
    assert sum(not c.is_authorized for c in cases) == 100
    assert all(extract_target_aliases(c.prompt) == [] for c in cases)


def test_historical_regression_suite_is_preserved():
    records, acl = _fixture()
    cases = generate_regression_cases(records, acl)
    assert len(cases) == 72
    assert sum(c.is_legitimate for c in cases) == 18
    assert sum(not c.is_authorized for c in cases) == 42
