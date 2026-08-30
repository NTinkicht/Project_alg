from healthrag.core import AuthorizationMatrix, PatientRecord
from healthrag.prompts import generate_cases


def test_prompt_suite_shape():
    records = [PatientRecord(f"PAT-{i:02d}", str(i), f"PatientAlias PAT-{i:02d}", f"C{i}", f"cond{i}", f"med{i}") for i in range(1, 26)]
    acl = AuthorizationMatrix({
        "A": [f"PAT-{i:02d}" for i in range(1, 6)],
        "B": [f"PAT-{i:02d}" for i in range(6, 11)],
        "C": [f"PAT-{i:02d}" for i in range(11, 16)],
        "D": [f"PAT-{i:02d}" for i in range(16, 21)],
        "E": [],
    })
    cases = generate_cases(records, acl)
    assert len(cases) == 72
    assert sum(c.is_legitimate for c in cases) == 18
    assert sum(not c.is_authorized for c in cases) == 42
    assert len({c.case_id for c in cases}) == 72
