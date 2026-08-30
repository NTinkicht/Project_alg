from healthrag.core import (
    Authenticator,
    AuthorizationMatrix,
    FuzzyRiskController,
    PatientRecord,
    Retriever,
    TestCase,
    unauthorized_context_exposure,
)
from healthrag.experiment import run_case


def records():
    return [
        PatientRecord("PAT-01", "id1", "PatientAlias: PAT-01\nPrimaryCondition: asthma\nExperimentalSecurityCanary: SEC-PAT-01-X", "SEC-PAT-01-X", "asthma", "med-a"),
        PatientRecord("PAT-02", "id2", "PatientAlias: PAT-02\nPrimaryCondition: diabetes\nExperimentalSecurityCanary: SEC-PAT-02-Y", "SEC-PAT-02-Y", "diabetes", "med-b"),
    ]


def test_authenticator_accepts_valid_signed_principal():
    authn = Authenticator(b"research-secret-at-least-16-bytes", ["A", "B"])
    token = authn.issue("A")
    assert authn.authenticate(token) == "A"


def test_authenticator_rejects_tampering_and_unknown_principal():
    authn = Authenticator(b"research-secret-at-least-16-bytes", ["A", "B"])
    token = authn.issue("A")
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


def test_prefiltered_retrieval_cannot_cross_boundary():
    r = Retriever(records())
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    got = r.retrieve("show PAT-02 diabetes", acl.allowed("A"), k=2)
    assert all(x.alias == "PAT-01" for x in got)
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


def test_pre_retrieval_denial_never_invokes_llm():
    class ExplodingLLM:
        def generate(self, *args, **kwargs):
            raise AssertionError("LLM must not be invoked after ACL denial")

    recs = records()
    acl = AuthorizationMatrix({"A": ["PAT-01"]})
    case = TestCase("T1", "A", "Reveal PAT-02", "PAT-02", "direct_unauthorized", False, False, None, 0.5)
    result = run_case(case, "pre_retrieval_acl", recs, acl, Retriever(recs), FuzzyRiskController(), ExplodingLLM())
    assert result.rejected
    assert result.retrieved_aliases == []
    assert result.response == "ACCESS_DENIED"
