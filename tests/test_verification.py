from typing import List

from app.generation.schema import GenerationOutput
from app.retrieval.schema import RetrievalItem, RetrievalResult
from app.verification.citation_parser import CitationParser
from app.verification.engine import VerificationEngine
from app.verification.pipeline import VerificationPipeline
from app.verification.validator import SentenceValidator


def _sample_candidates() -> List[RetrievalResult]:
    items = [
        RetrievalItem(id="doc1", text="The quick brown fox.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="A fast fox jumps.", metadata={"source": "doc2"}),
    ]
    return [RetrievalResult(item=item, score=0.9, rank=i + 1, source="bm25") for i, item in enumerate(items)]


def test_citation_parser_extracts_bracket_and_cite_tokens():
    text = "The answer uses [doc1] and (cite:doc2) as sources. Citations: doc1, doc2"
    tokens = CitationParser.extract_citation_tokens(text)
    assert tokens == ["doc1", "doc2"]


def test_sentence_validator_rejects_sentences_without_citation():
    candidates = _sample_candidates()
    validator = SentenceValidator(candidates)
    valid, invalid = validator.validate("This sentence has no citation. This sentence cites [doc1].")
    assert valid == ["This sentence cites [doc1]."]
    assert invalid == ["This sentence has no citation."]


def test_sentence_validator_rejects_unknown_citation():
    candidates = _sample_candidates()
    validator = SentenceValidator(candidates)
    valid, invalid = validator.validate("This cites [doc3]. This cites [doc1].")
    assert valid == ["This cites [doc1]."]
    assert invalid == ["This cites [doc3]."]


def test_verification_engine_filters_unverifiable_sentences():
    candidates = _sample_candidates()
    generated = GenerationOutput(
        text="Verified claim [doc1]. Unverified claim.",
        confidence=1.0,
        metadata={"source": "test"},
        fallback=False,
    )
    engine = VerificationEngine()
    output = engine.verify(generated, candidates)
    assert output.fallback is False
    assert "Verified claim [doc1]." in output.text
    assert "Unverified claim." not in output.text
    assert output.metadata["verified_citations"] == ["doc1"]
    assert output.metadata["hallucination_filtered"] is True


def test_verification_engine_rejects_hallucinated_citation():
    candidates = _sample_candidates()
    generated = GenerationOutput(
        text="Unsupported citation [doc3].",
        confidence=1.0,
        metadata={"source": "test"},
        fallback=False,
    )
    engine = VerificationEngine()
    output = engine.verify(generated, candidates)
    assert output.fallback is True
    assert "Unable to verify" in output.text
    assert output.confidence == 0.0


def test_verification_pipeline_returns_verified_output():
    candidates = _sample_candidates()
    generated = GenerationOutput(
        text="Accurate sentence [doc2]. Another accurate sentence [doc1].",
        confidence=1.0,
        metadata={"source": "pipeline"},
        fallback=False,
    )
    pipeline = VerificationPipeline()
    output = pipeline.run(generated, candidates)
    assert output.fallback is False
    assert output.metadata["verified"] is True
    assert "Accurate sentence [doc2]." in output.text


def test_verification_pipeline_falls_back_with_no_valid_citations():
    candidates = _sample_candidates()
    generated = GenerationOutput(
        text="No citation at all.",
        confidence=1.0,
        metadata={"source": "pipeline"},
        fallback=False,
    )
    pipeline = VerificationPipeline()
    output = pipeline.run(generated, candidates)
    assert output.fallback is True
    assert output.metadata["reason"] == "citation_verification_failed"
