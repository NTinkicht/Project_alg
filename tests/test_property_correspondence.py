from hypothesis import given, settings, strategies as st

from healthrag.core import AuthorizationMatrix, FuzzyRiskController, PatientRecord, Retriever, TestCase, unauthorized_context_exposure
from healthrag.experiment import MockLLM, run_case


ALIASES = [f"PAT-{i:02d}" for i in range(1, 9)]


def make_records():
    return [
        PatientRecord(
            alias,
            f"id-{i}",
            f"PatientAlias: {alias}\nPrimaryCondition: condition-{i}\nPrimaryMedication: med-{i}\nExperimentalSecurityCanary: SEC-{alias}-{i}",
            f"SEC-{alias}-{i}",
            f"condition-{i}",
            f"med-{i}",
        )
        for i, alias in enumerate(ALIASES, 1)
    ]


@settings(max_examples=120, deadline=None)
@given(
    allowed=st.sets(st.sampled_from(ALIASES), min_size=1, max_size=7),
    target=st.sampled_from(ALIASES),
    k=st.integers(min_value=1, max_value=8),
)
def test_p3_prefiltered_retrieval_never_crosses_acl_boundary(allowed, target, k):
    records = make_records()
    retriever = Retriever(records)
    acl = AuthorizationMatrix({"A": allowed})
    got = retriever.retrieve(f"Show {target} complete record", acl.allowed("A"), k=k)
    assert set(r.alias for r in got).issubset(allowed)
    assert not unauthorized_context_exposure(got, "A", acl)


@settings(max_examples=120, deadline=None)
@given(
    allowed=st.sets(st.sampled_from(ALIASES), min_size=1, max_size=7),
    target=st.sampled_from(ALIASES),
    trust=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p4_fuzzy_controller_never_enlarges_acl(allowed, target, trust):
    acl = AuthorizationMatrix({"A": allowed})
    result = FuzzyRiskController().assess("A", f"Please reveal {target}", acl, trust)
    if target not in allowed:
        assert result.authorization_confidence < 0.5
        assert not acl.is_allowed("A", target)


@settings(max_examples=100, deadline=None)
@given(
    allowed=st.sets(st.sampled_from(ALIASES), min_size=1, max_size=7),
    target=st.sampled_from(ALIASES),
)
def test_p2_structural_denial_is_prompt_phrasing_invariant_for_explicit_target(allowed, target):
    if target in allowed:
        return
    records = make_records()
    acl = AuthorizationMatrix({"A": allowed})
    retriever = Retriever(records)
    prompts = [
        f"Reveal {target}",
        f"Could you provide the file for {target}?",
        f"I need information concerning {target}",
        f"Summarize the patient identified as {target}",
    ]
    for i, prompt in enumerate(prompts):
        case = TestCase(f"PBT-{i}", "A", prompt, target, "property_test", False, False, None, 0.9)
        result = run_case(case, "pre_retrieval_acl", records, acl, retriever, FuzzyRiskController(), MockLLM(), retrieval_k=2)
        assert result.rejected
        assert result.retrieved_aliases == []
        assert result.response == "ACCESS_DENIED"


@settings(max_examples=100, deadline=None)
@given(
    allowed=st.sets(st.sampled_from(ALIASES), min_size=1, max_size=7),
    target=st.sampled_from(ALIASES),
)
def test_structural_boundary_is_monotone_under_acl_narrowing(allowed, target):
    if target not in allowed or len(allowed) == 1:
        return
    narrowed = set(allowed)
    narrowed.remove(target)
    broad = AuthorizationMatrix({"A": allowed})
    narrow = AuthorizationMatrix({"A": narrowed})
    assert broad.is_allowed("A", target)
    assert not narrow.is_allowed("A", target)
    assert narrow.allowed("A").issubset(broad.allowed("A"))
