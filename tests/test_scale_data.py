from healthrag.core import PatientRecord
from healthrag.scale_data import _proportional_acl, _relabel


def _record(i: int) -> PatientRecord:
    alias = f"PAT-{i:04d}" if i <= 1000 else f"PAT-{i}"
    return PatientRecord(alias, f"id-{i}", f"PatientAlias: {alias}\nExperimentalSecurityCanary: old", "old", "c", "m")


def test_relabel_recomputes_alias_and_canary():
    r = _relabel(_record(1001), "PAT-5001")
    assert r.alias == "PAT-5001"
    assert "PatientAlias: PAT-5001" in r.text
    assert r.canary in r.text
    assert r.canary != "old"


def test_proportional_acl_is_nested_and_exact_at_10k():
    records = [_record(i) for i in range(1, 10001)]
    base = {
        "A": [r.alias for r in records[0:200]],
        "B": [r.alias for r in records[200:400]],
        "C": [r.alias for r in records[400:600]],
        "D": [r.alias for r in records[600:800]],
        "E": [],
    }
    mapping, protected = _proportional_acl(base, records, 10000)
    assert {a: len(mapping[a]) for a in "ABCD"} == {"A": 2000, "B": 2000, "C": 2000, "D": 2000}
    assert len(protected) == 2000
    assert all(alias in mapping["A"] for alias in base["A"])
