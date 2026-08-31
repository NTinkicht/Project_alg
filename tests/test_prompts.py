from healthrag.core import AuthorizationMatrix, PatientRecord
from healthrag.prompts import generate_cases, generate_regression_cases


def _fixture(n=1000):
    width = max(2, len(str(n)))
    records = [PatientRecord(f"PAT-{i:0{width}d}", str(i), f"PatientAlias PAT-{i:0{width}d}", f"C{i}", f"cond{i}", f"med{i}") for i in range(1, n + 1)]
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


def test_historical_regression_suite_is_preserved():
    records, acl = _fixture()
    cases = generate_regression_cases(records, acl)
    assert len(cases) == 72
    assert sum(c.is_legitimate for c in cases) == 18
    assert sum(not c.is_authorized for c in cases) == 42
