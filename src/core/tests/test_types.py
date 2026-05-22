"""Tests for aap_core.types module - AgentMessage, BaseChain, BaseLLMChain, TokenUsage."""

import json

import pytest
from aap_core.types import (
    AgentMessage,
    BaseChain,
    BaseLLMChain,
    TokenUsage,
)


class TestTokenUsage:
    """Tests for TokenUsage TypedDict."""

    def test_token_usage_creation(self):
        """Test creating a TokenUsage dict."""
        usage: TokenUsage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_token_usage_zero_values(self):
        """Test TokenUsage with zero values."""
        usage: TokenUsage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        assert usage["total_tokens"] == 0

    def test_token_usage_large_values(self):
        """Test TokenUsage with large values."""
        usage: TokenUsage = {
            "input_tokens": 1000000,
            "output_tokens": 500000,
            "total_tokens": 1500000,
        }
        assert usage["total_tokens"] == 1500000


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_construction_required_fields(self):
        """Test AgentMessage construction with required fields."""
        msg = AgentMessage(query="test query")
        assert msg.query == "test query"
        assert msg.responses == []
        assert msg.context is None
        assert msg.execution_result is None

    def test_construction_with_all_fields(self):
        """Test AgentMessage construction with all fields."""
        msg = AgentMessage(
            query="test query",
            query_media=[("text", "hello")],
            origin="test_agent",
            responses=[("agent1", "response1")],
            context={"key": "value"},
            execution_result="success",
            error_message=None,
            media=[("image", "base64data")],
            token_usage={
                "total": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            },
        )
        assert msg.query == "test query"
        assert msg.origin == "test_agent"
        assert msg.execution_result == "success"
        assert len(msg.responses) == 1

    def test_flatten_dict_simple(self):
        """Test flatten_dict with simple nested dict."""
        msg = AgentMessage(query="test")
        nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        result = msg.flatten_dict(nested)
        assert result == {"a": 1, "b_c": 2, "b_d_e": 3}

    def test_flatten_dict_empty(self):
        """Test flatten_dict with empty dict."""
        msg = AgentMessage(query="test")
        result = msg.flatten_dict({})
        assert result == {}

    def test_flatten_dict_with_custom_separator(self):
        """Test flatten_dict with custom separator."""
        msg = AgentMessage(query="test")
        nested = {"a": {"b": 1}}
        result = msg.flatten_dict(nested, sep="-")
        assert result == {"a-b": 1}

    def test_flatten_dict_no_parent_key(self):
        """Test flatten_dict without parent_key."""
        msg = AgentMessage(query="test")
        nested = {"a": 1, "b": 2}
        result = msg.flatten_dict(nested)
        assert result == {"a": 1, "b": 2}

    def test_to_dict_without_context(self):
        """Test to_dict without context."""
        msg = AgentMessage(query="test query", origin="test_agent")
        result = msg.to_dict()
        assert result["query"] == "test query"
        assert result["origin"] == "test_agent"
        assert "context" not in result

    def test_to_dict_with_context(self):
        """Test to_dict with context."""
        msg = AgentMessage(
            query="test query",
            context={"a": 1, "b": {"c": 2}},
        )
        result = msg.to_dict()
        assert result["query"] == "test query"
        assert result["context_a"] == 1
        assert result["context_b_c"] == 2

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        msg = AgentMessage(query="test query", execution_result=None)
        result = msg.to_dict()
        assert "execution_result" not in result

    def test_dump_json(self):
        """Test dump_json produces valid JSON string."""
        msg = AgentMessage(query="test query", origin="test_agent")
        json_str = msg.dump_json()
        parsed = json.loads(json_str)
        assert parsed["query"] == "test query"
        assert parsed["origin"] == "test_agent"

    def test_dump_json_with_context(self):
        """Test dump_json with context."""
        msg = AgentMessage(
            query="test query",
            context={"key": "value"},
        )
        json_str = msg.dump_json()
        parsed = json.loads(json_str)
        assert parsed["context_key"] == "value"

    def test_model_copy(self):
        """Test AgentMessage model_copy."""
        msg = AgentMessage(query="test query", responses=[("a", "b")])
        copied = msg.model_copy(deep=True)
        assert copied.query == msg.query
        assert copied.responses == msg.responses
        assert copied.responses is not msg.responses  # deep copy


class TestBaseChain:
    """Tests for BaseChain abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseChain cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChain()


class MockLLMChain(BaseLLMChain):
    """Mock implementation of BaseLLMChain for testing."""

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message.responses.append(("mock_chain", "mock response"))
        message.execution_result = "success"
        return message


class TestBaseLLMChain:
    """Tests for BaseLLMChain abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseLLMChain cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMChain()

    def test_construction_with_name(self):
        """Test BaseLLMChain construction with custom name."""
        chain = MockLLMChain(name="custom_chain")
        assert chain.name == "custom_chain"

    def test_construction_default_name(self):
        """Test BaseLLMChain construction with default name."""
        chain = MockLLMChain()
        assert chain.name == "chain"

    def test_invoke(self):
        """Test BaseLLMChain invoke method."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        result = chain.invoke(msg)
        assert result.execution_result == "success"
        assert len(result.responses) == 1

    def test_ainvoke(self):
        """Test BaseLLMChain ainvoke method (sync version calls invoke)."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        # ainvoke is async, so we call invoke directly for sync testing
        result = chain.invoke(msg)
        assert result.execution_result == "success"

    def test_call(self):
        """Test BaseLLMChain __call__ method."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        result = chain(msg)
        assert result.execution_result == "success"

    def test_call_with_kwargs(self):
        """Test BaseLLMChain __call__ with kwargs."""
        chain = MockLLMChain(name="test_chain")
        msg = AgentMessage(query="test")
        result = chain(msg, extra_kwarg="value")
        assert result.execution_result == "success"
