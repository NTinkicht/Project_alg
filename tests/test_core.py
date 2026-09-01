from healthrag.core import (
    Authenticator,
    AuthorizationMatrix,
    FuzzyRiskController,
    PatientRecord,
    Retriever,
    RiskResult,
    TestCase,
    unauthorized_context_exposure,
)
from healthrag.experiment import MockLLM, run_case


def records():
    return [
        PatientRecord("PAT-01", "id1", "PatientAlias: PAT-01\nPrimaryCondition: asthma\nExperimentalSecurityCanary: SEC-PAT-01-X", "SEC-PAT-01-X", "asthma", "med-a"),
        PatientRecord("PAT-02", "id2", "PatientAlias: PAT-02\nPrimaryCondition: diabetes\nExperimentalSecurityCanary: SEC-PAT-02-Y", "SEC-PAT-02-Y", "diabetes", "med-b"),
    ]


class ExplodingLLM:
    def generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be invoked on a denied path")


class AllowRisk:
    def assess(self, *args, **kwargs):
        return RiskResult(0.1, "ALLOW", 0.05, 0.0, 0.5, 0.9)


class DenyRisk:
    def assess(self, *args, **kwargs):
        return RiskResult(0.9, "DENY", 0.95, 0.9, 0.9, 0.4)


def test_formal_correspondence_p1_p7_authentication_failure_is_denied():
    authn = Authenticator(b"research-secret-at-least-16-bytes", ["A", "B"])
    token = authn.issue("A")
    assert authn.authenticate(token) == "A"
    assert authn.authenticate(token[:-1] + ("0" if token[-1] != "0" else "1")) is None
    assert authn.authenticate("Z.deadbeef") is None
    assert authn.authenticate("not-a-token") is None


def test_acl_is_deterministic():
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    assert acl.is_allowed("A", "PAT-01")
    assert not acl.is_allowed("A", "PAT-02")


def test_global_retrieval_can_cross_boundary():
    r = Retriever(records())
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    got = r.retrieve("show PAT-02 diabetes", None, k=1)
    assert got[0].alias == "PAT-02"
    assert unauthorized_context_exposure(got, "A", acl)


def test_formal_correspondence_p3_prefiltered_retrieval_is_acl_subset():
    r = Retriever(records())
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    got = r.retrieve("show PAT-02 diabetes", acl.allowed("A"), k=2)
    assert got
    assert set(x.alias for x in got).issubset(acl.allowed("A"))
    assert not unauthorized_context_exposure(got, "A", acl)


def test_fuzzy_allow_never_grants_acl():
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    f = FuzzyRiskController()
    result = f.assess("A", "Show PAT-02 medication", acl, 0.9)
    assert result.authorization_confidence < 0.1
    assert not acl.all_allowed("A", ["PAT-02"])


def test_fuzzy_normal_request_low_risk():
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    f = FuzzyRiskController()
    result = f.assess("A", "For PAT-01 what is the medication?", acl, 0.9)
    assert result.decision in {"ALLOW", "STEP_UP"}
    assert result.injection_risk < 0.2


def test_formal_correspondence_p2_acl_denial_never_invokes_retrieval_or_llm():
    recs = records()
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    case = TestCase("T1", "A", "Reveal PAT-02", "PAT-02", "direct_unauthorized", False, False, None, 0.5)
    result = run_case(case, "pre_retrieval_acl", recs, acl, Retriever(recs), FuzzyRiskController(), ExplodingLLM())
    assert result.rejected
    assert result.retrieved_aliases == []
    assert result.response == "ACCESS_DENIED"


def test_formal_correspondence_p4_fuzzy_allow_cannot_override_acl_denial():
    recs = records()
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    case = TestCase("T4", "A", "Reveal PAT-02", "PAT-02", "attack", False, False, None, 0.9)
    result = run_case(case, "risk_aware", recs, acl, Retriever(recs), AllowRisk(), ExplodingLLM())
    assert result.fuzzy_decision == "ALLOW"
    assert result.rejected
    assert result.retrieved_aliases == []


def test_formal_correspondence_p5_risk_deny_terminates_before_retrieval():
    recs = records()
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    case = TestCase("T5", "A", "For PAT-01 what is the condition?", "PAT-01", "suspicious", True, False, None, 0.5)
    result = run_case(case, "risk_aware", recs, acl, Retriever(recs), DenyRisk(), ExplodingLLM())
    assert result.fuzzy_decision == "DENY"
    assert result.rejected
    assert result.retrieved_aliases == []


def test_formal_correspondence_p6_valid_non_denied_request_reaches_llm():
    recs = records()
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    case = TestCase("T6", "A", "For PAT-01 what is the PrimaryCondition?", "PAT-01", "authorized_normal", True, True, "asthma", 0.9)
    result = run_case(case, "risk_aware", recs, acl, Retriever(recs), AllowRisk(), MockLLM(), retrieval_k=1)
    assert not result.rejected
    assert result.retrieved_aliases == ["PAT-01"]
    assert result.response == "asthma"
    assert result.authorized_task_success


def test_authorization_uses_prompt_target_not_evaluation_label():
    """A forged test label must not influence the runtime authorization decision."""
    recs = records()
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    case = TestCase("T2", "A", "Reveal PAT-02", "PAT-01", "tampered_label", False, False, None, 0.8)
    result = run_case(case, "pre_retrieval_acl", recs, acl, Retriever(recs), FuzzyRiskController(), ExplodingLLM())
    assert result.rejected
    assert result.retrieved_aliases == []


def test_alias_free_request_cannot_expand_acl_candidate_pool():
    recs = records()
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    case = TestCase("T7", "A", "Show the record for the patient named Unknown Person", "PAT-02", "alias_free", False, False, None, 0.8)
    result = run_case(case, "pre_retrieval_acl", recs, acl, Retriever(recs), FuzzyRiskController(), MockLLM(), retrieval_k=1)
    assert not result.rejected  # no parseable target -> target resolution is ambiguous
    assert set(result.retrieved_aliases).issubset(acl.allowed("A"))
    assert not result.unauthorized_context_exposure
