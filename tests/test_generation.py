"""
Tests for Phase 2: citation enforcement, prompt manager, and answer generator.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.citation_utils import CitationEnforcer, CitationResult, _count_uncited_sentences
from src.utils.prompt_manager import PromptManager
from src.generation.llm_client import AnthropicClient
from src.generation.answer_generator import AnswerGenerator, AnswerResult


# ------------------------------------------------------------------ #
#  _count_uncited_sentences                                            #
# ------------------------------------------------------------------ #

def test_count_uncited_all_cited():
    text = "Hypertension requires treatment [1]. Diet is important [2]."
    assert _count_uncited_sentences(text) == 0


def test_count_uncited_some_uncited():
    text = "Hypertension requires treatment. Diet is important [2]."
    count = _count_uncited_sentences(text)
    assert count >= 1


def test_count_uncited_disclaimer_not_counted():
    text = "⚠ DISCLAIMER: Always consult a healthcare professional."
    assert _count_uncited_sentences(text) == 0


# ------------------------------------------------------------------ #
#  CitationEnforcer                                                    #
# ------------------------------------------------------------------ #

def test_citation_enforcer_valid_response(sample_retrieved_chunks):
    enforcer = CitationEnforcer(max_uncited_sentences=1)
    answer = (
        "Hypertension is treated with thiazide diuretics [1]. "
        "ACE inhibitors are preferred in diabetic patients [2]. "
        "Calcium channel blockers help elderly patients [3]."
    )
    result = enforcer.enforce(answer, sample_retrieved_chunks)
    assert isinstance(result, CitationResult)
    assert result.declined is False
    assert len(result.citations) == 3


def test_citation_enforcer_insufficient_context(sample_retrieved_chunks):
    enforcer = CitationEnforcer()
    answer = "INSUFFICIENT_CONTEXT: No information available about this topic."
    result = enforcer.enforce(answer, sample_retrieved_chunks)
    assert result.declined is True
    assert result.is_valid is False


def test_citation_enforcer_out_of_range_citation(sample_retrieved_chunks):
    enforcer = CitationEnforcer()
    answer = "Treatment includes beta-blockers [99]."  # [99] out of range
    result = enforcer.enforce(answer, sample_retrieved_chunks)
    assert 99 in result.missing_citation_numbers
    assert result.is_valid is False


def test_citation_enforcer_context_block_format(sample_retrieved_chunks):
    enforcer = CitationEnforcer()
    block = enforcer.build_context_block(sample_retrieved_chunks)
    assert "[1]" in block
    assert "[2]" in block
    assert "[3]" in block
    for chunk in sample_retrieved_chunks:
        assert chunk.text in block


def test_citation_enforcer_formatted_references(sample_retrieved_chunks):
    enforcer = CitationEnforcer()
    answer = "Treatment [1] is effective [2]."
    result = enforcer.enforce(answer, sample_retrieved_chunks)
    refs = result.formatted_references()
    assert "[1]" in refs
    assert "[2]" in refs


# ------------------------------------------------------------------ #
#  PromptManager                                                       #
# ------------------------------------------------------------------ #

def test_prompt_manager_loads_prompts():
    pm = PromptManager()
    prompts = pm.list_prompts()
    assert "healthcare_rag_system" in prompts
    assert "healthcare_rag_user" in prompts
    assert "decline_response" in prompts


def test_prompt_manager_get_no_variables():
    pm = PromptManager()
    system = pm.get("healthcare_rag_system")
    assert isinstance(system, str)
    assert len(system) > 50


def test_prompt_manager_get_with_variables():
    pm = PromptManager()
    user = pm.get(
        "healthcare_rag_user",
        context="[1] Test chunk text",
        question="What is hypertension?",
    )
    assert "What is hypertension?" in user
    assert "[1] Test chunk text" in user


def test_prompt_manager_missing_variable():
    pm = PromptManager()
    with pytest.raises(KeyError):
        pm.get("healthcare_rag_user")  # missing context and question


def test_prompt_manager_version():
    pm = PromptManager()
    v = pm.version("healthcare_rag_system")
    assert isinstance(v, str)


def test_prompt_manager_key_not_found():
    pm = PromptManager()
    with pytest.raises(KeyError):
        pm.get("nonexistent_prompt_xyz")


# ------------------------------------------------------------------ #
#  AnthropicClient (mocked)                                            #
# ------------------------------------------------------------------ #

def test_anthropic_client_complete():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        client = AnthropicClient(model="claude-opus-4-6")

        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mocked answer [1].")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=20)
        mock_anthropic.messages.create.return_value = mock_response
        client._client = mock_anthropic

        text, input_tokens, output_tokens = client.complete("system prompt", "user prompt")
        assert text == "Mocked answer [1]."
        assert input_tokens == 100
        assert output_tokens == 20


# ------------------------------------------------------------------ #
#  AnswerGenerator (integration-style, fully mocked)                   #
# ------------------------------------------------------------------ #

def test_answer_generator_end_to_end(sample_retrieved_chunks):
    # Mock dependencies
    mock_llm = MagicMock()
    mock_llm.model = "claude-opus-4-6"
    mock_llm.complete.return_value = (
        "Thiazide diuretics are first-line treatment for hypertension [1]. "
        "ACE inhibitors are preferred in diabetic patients [2].",
        350,   # input_tokens
        80,    # output_tokens
    )

    mock_retriever = MagicMock()
    mock_retriever.query.return_value = sample_retrieved_chunks

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = sample_retrieved_chunks[:2]

    pm = PromptManager()
    enforcer = CitationEnforcer(max_uncited_sentences=1)

    generator = AnswerGenerator(
        llm_client=mock_llm,
        retriever=mock_retriever,
        reranker=mock_reranker,
        prompt_manager=pm,
        citation_enforcer=enforcer,
        decline_on_invalid_citations=False,  # allow uncited for test
    )

    result = generator.answer("What treats hypertension?")
    assert isinstance(result, AnswerResult)
    assert result.question == "What treats hypertension?"
    assert not result.declined
    assert len(result.retrieved_chunks) > 0


def test_answer_generator_declines_on_no_context():
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.query.return_value = []  # no results

    generator = AnswerGenerator(
        llm_client=mock_llm,
        retriever=mock_retriever,
        reranker=None,
        prompt_manager=PromptManager(),
    )
    result = generator.answer("What is the cure for everything?")
    assert result.declined is True


def test_answer_generator_declines_on_insufficient_context(sample_retrieved_chunks):
    mock_llm = MagicMock()
    mock_llm.model = "claude-opus-4-6"
    mock_llm.complete.return_value = (
        "INSUFFICIENT_CONTEXT: The documents do not address this topic.",
        200,   # input_tokens
        15,    # output_tokens
    )
    mock_retriever = MagicMock()
    mock_retriever.query.return_value = sample_retrieved_chunks
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = sample_retrieved_chunks

    generator = AnswerGenerator(
        llm_client=mock_llm,
        retriever=mock_retriever,
        reranker=mock_reranker,
        prompt_manager=PromptManager(),
    )
    result = generator.answer("Some unanswerable question")
    assert result.declined is True
