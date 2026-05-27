import asyncio
import tempfile
from pathlib import Path

import pytest

from app.generation.client import OllamaClient, VLLMClient
from app.generation.guard import ConfidenceGuard
from app.generation.pipeline import InferencePipeline
from app.generation.prompt import PromptBuilder, escape_xml
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import BGEM3EmbeddingService
from app.retrieval.schema import RetrievalItem, RetrievalResult
from app.retrieval.vector_store import VectorStore, VectorRetriever


def _sample_items():
    return [
        RetrievalItem(id="doc1", text="The quick brown <fox> jumps over the lazy dog.", metadata={"source": "doc1"}),
        RetrievalItem(id="doc2", text="A fast brown fox leaps across sleepy canines.", metadata={"source": "doc2"}),
    ]


def test_escape_xml_encodes_xml_characters():
    raw = "<tag> & "
    encoded = escape_xml(raw)
    assert "<" not in encoded
    assert ">" not in encoded
    assert "&amp;" in encoded


def test_prompt_builder_creates_isolated_xml_context():
    items = _sample_items()
    results = [RetrievalResult(item=item, score=0.5, rank=index + 1, source="bm25") for index, item in enumerate(items)]
    prompt = PromptBuilder().build_prompt("What is the topic?", results)
    assert "<System>" in prompt
    assert "<Context>" in prompt
    assert "<Source id=\"doc1\">" in prompt
    assert "<Query>What is the topic?</Query>" in prompt
    assert "<tag>" not in prompt


def test_ollama_client_is_deterministic_and_requires_exact_settings():
    client = OllamaClient(model="qwen2.5-5b-instruct")
    output_a = client.generate("prompt text")
    output_b = client.generate("prompt text")
    assert output_a.text == output_b.text
    assert output_a.metadata["runtime"] == "ollama"
    with pytest.raises(ValueError):
        OllamaClient(model="qwen2.5-5b-instruct", temperature=0.5)


def test_vllm_client_is_deterministic_and_requires_exact_settings():
    client = VLLMClient(model="qwen2.5-5b-instruct")
    output = client.generate("prompt text")
    assert "Deterministic vLLM output" in output.text
    with pytest.raises(ValueError):
        VLLMClient(model="qwen2.5-5b-instruct", top_p=0.8)


def test_confidence_guard_triggers_fallback_on_low_scores():
    items = _sample_items()
    results = [RetrievalResult(item=items[0], score=0.1, rank=1, source="bm25")]
    guard = ConfidenceGuard(threshold=0.35)
    assert not guard.is_confident(results)
    assert guard.fallback() == "Unable to answer with sufficient confidence from the provided context."


def test_inference_pipeline_falls_back_when_retrieval_confidence_is_low():
    items = _sample_items()
    results = [RetrievalResult(item=items[0], score=0.1, rank=1, source="bm25")]
    client = OllamaClient()
    pipeline = InferencePipeline(client=client)
    output = pipeline.infer("What is the topic?", results)
    assert output.fallback is True
    assert "Unable to answer" in output.text
    assert output.confidence == 0.0


def test_inference_pipeline_generates_with_context_and_prompt_key():
    items = _sample_items()
    bm25 = BM25Retriever(items)
    query = "What does the text describe?"
    results = bm25.retrieve(query, top_k=2)
    client = VLLMClient()
    pipeline = InferencePipeline(client=client)
    output = pipeline.infer(query, results)
    assert output.fallback is False
    assert output.confidence == 1.0
    assert output.metadata.get("prompt_key") is not None
