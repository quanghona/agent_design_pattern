"""Tests for aap_core.utils module."""

from aap_core.utils import remove_thinking


class TestRemoveThinking:
    """Tests for the remove_thinking utility function."""

    def test_remove_thinking_tags(self):
        """Test removing thinking tags from a response."""
        response = "<think>Let me think about this.</think>Hello world"
        result = remove_thinking(response)
        assert result == "Hello world"

    def test_remove_thinking_no_tags(self):
        """Test response without thinking tags is unchanged."""
        response = "Hello world"
        result = remove_thinking(response)
        assert result == "Hello world"

    def test_remove_thinking_missing_opening_tag_qwen(self):
        """Test Qwen model case: has </think> but no <think> opening."""
        response = "<think>Let me think.</think>Hello world"
        result = remove_thinking(response)
        assert "<think>" not in result
        assert "Hello world" in result

    def test_remove_thinking_empty_response(self):
        """Test with empty string."""
        result = remove_thinking("")
        assert result == ""

    def test_remove_thinking_multiline_thinking(self):
        """Test removing thinking with multiline content."""
        response = "<think>Line 1\nLine 2\nLine 3</think>Final answer"
        result = remove_thinking(response)
        assert "Line 1" not in result
        assert "Line 2" not in result
        assert "Final answer" in result

    def test_remove_thinking_only_closing_tag(self):
        """Test with only closing tag present."""
        response = "</think>Hello"
        result = remove_thinking(response)
        assert result == "Hello"

    def test_remove_thinking_only_opening_tag(self):
        """Test with only opening tag present (no closing)."""
        response = "<think>Hello"
        result = remove_thinking(response)
        # Should not crash, returns as-is since no closing tag
        assert "<think>" in result

    def test_remove_thinking_multiple_thinking_blocks(self):
        """Test with multiple thinking blocks (only first is removed by DOTALL)."""
        response = "<think>First</think>Middle<think>Second</think>End"
        result = remove_thinking(response)
        # DOTALL removes from first <think> to last </think>
        assert "First" not in result
        assert "Second" not in result
        assert "End" in result
