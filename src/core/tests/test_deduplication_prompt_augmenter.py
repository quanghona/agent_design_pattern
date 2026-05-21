import pytest

from aap_core.prompt_augmenter import DeduplicationPromptAugmenter
from aap_core.types import AgentMessage


class TestDeduplicationPromptAugmenter:
    """Tests for DeduplicationPromptAugmenter using rensa C-MinHash."""

    @pytest.fixture
    def augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.8}
        )

    @pytest.fixture
    def low_threshold_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.3}
        )

    def test_constructor_custom_params(self):
        """Test constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 123, "threshold": 0.7}
        )
        assert aug._algo_config.num_perm == 64
        assert aug._algo_config.seed == 123
        assert aug._algo_config.threshold == 0.7
        assert aug._algo_config.algorithm_name == "minhash"

    def test_constructor_invalid_threshold_too_low(self):
        """Test that threshold below 0.0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"threshold": -0.1})

    def test_constructor_invalid_threshold_too_high(self):
        """Test that threshold above 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"threshold": 1.1})

    def test_constructor_invalid_threshold_boundary_zero(self):
        """Test that threshold=0.0 is accepted (boundary)."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.0}
        )
        assert aug._algo_config.threshold == 0.0

    def test_constructor_invalid_threshold_boundary_one(self):
        """Test that threshold=1.0 is accepted (boundary)."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 1.0}
        )
        assert aug._algo_config.threshold == 1.0

    def test_constructor_invalid_num_perm_zero(self):
        """Test that num_perm=0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={"num_perm": 0, "seed": 42, "threshold": 0.8}
            )

    def test_constructor_invalid_num_perm_negative(self):
        """Test that num_perm<0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={"num_perm": -10, "seed": 42, "threshold": 0.8}
            )

    def test_constructor_invalid_seed_negative(self):
        """Test that seed<0 raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={"num_perm": 64, "seed": -1, "threshold": 0.8}
            )

    def test_constructor_invalid_algorithm_name(self):
        """Test that unsupported algorithm_name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "unsupported_algo"}
            )

    def test_constructor_simhash_not_implemented(self):
        """Test that simhash algorithm raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "simhash", "threshold": 0.8}
            )

    def test_constructor_invalid_threshold_type(self):
        """Test that non-coercible threshold type raises ValidationError."""
        from pydantic import ValidationError

        # Pydantic v2 coerces "0.8" to 0.8, so test with a list which can't be coerced
        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": [0.8]})

    def test_constructor_invalid_num_perm_type(self):
        """Test that non-integer num_perm raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"num_perm": 64.5})

    def test_constructor_invalid_seed_type(self):
        """Test that non-integer seed raises ValueError."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(algo_args={"seed": 42.5})

    def test_constructor_nested_dict_in_algo_args(self):
        """Test that nested dict values in algo_args raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": {"nested": "dict"}})

    def test_constructor_list_in_algo_args(self):
        """Test that list values in algo_args raise ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": [0.8, 0.9]})

    def test_constructor_none_value_for_threshold(self):
        """Test that None value for threshold raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"threshold": None})

    def test_constructor_none_value_for_num_perm(self):
        """Test that None value for num_perm raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"num_perm": None})

    def test_constructor_none_value_for_seed(self):
        """Test that None value for seed raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(algo_args={"seed": None})

    def test_constructor_empty_algo_args(self):
        """Test that empty algo_args uses defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={})
        assert aug._algo_config.num_perm == 128
        assert aug._algo_config.seed == 42
        assert aug._algo_config.threshold == 0.9
        assert aug._algo_config.algorithm_name == "minhash"

    def test_constructor_partial_algo_args(self):
        """Test that partial algo_args fills in defaults for missing keys."""
        aug = DeduplicationPromptAugmenter(algo_args={"threshold": 0.5})
        assert aug._algo_config.threshold == 0.5
        assert aug._algo_config.num_perm == 128  # default
        assert aug._algo_config.seed == 42  # default
        assert aug._algo_config.algorithm_name == "minhash"

    def test_constructor_algorithm_name_with_other_params(self):
        """Test that explicit algorithm_name=minhash works with other params."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash",
                "threshold": 0.9,
                "num_perm": 32,
                "seed": 100,
            }
        )
        assert aug._algo_config.algorithm_name == "minhash"
        assert aug._algo_config.threshold == 0.9
        assert aug._algo_config.num_perm == 32
        assert aug._algo_config.seed == 100

    # --- LSH Algorithm Tests ---

    @pytest.fixture
    def lsh_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.8,
                "num_bands": 16,
            }
        )

    @pytest.fixture
    def lsh_low_threshold_augmenter(self):
        return DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.3,
                "num_bands": 16,
            }
        )

    def test_constructor_lsh_defaults(self):
        """Test that LSH constructor uses correct defaults."""
        aug = DeduplicationPromptAugmenter(algo_args={"algorithm_name": "minhash_lsh"})
        assert aug._algo_config.algorithm_name == "minhash_lsh"
        assert aug._algo_config.num_perm == 128
        assert aug._algo_config.seed == 42
        assert aug._algo_config.threshold == 0.9
        assert aug._algo_config.num_bands == 16  # default, 128/16=8

    def test_constructor_lsh_custom_params(self):
        """Test LSH constructor with custom parameters."""
        aug = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 123,
                "threshold": 0.7,
                "num_bands": 16,
            }
        )
        assert aug._algo_config.algorithm_name == "minhash_lsh"
        assert aug._algo_config.num_perm == 64
        assert aug._algo_config.seed == 123
        assert aug._algo_config.threshold == 0.7
        assert aug._algo_config.num_bands == 16

    def test_constructor_lsh_invalid_num_bands_zero(self):
        """Test that num_bands=0 raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "num_bands": 0,
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                }
            )

    def test_constructor_lsh_invalid_num_bands_negative(self):
        """Test that num_bands<0 raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "num_bands": -5,
                    "num_perm": 64,
                    "seed": 42,
                    "threshold": 0.8,
                }
            )

    def test_constructor_lsh_invalid_threshold(self):
        """Test that invalid threshold raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "threshold": 1.5,
                    "num_perm": 64,
                    "seed": 42,
                    "num_bands": 16,
                }
            )

    def test_constructor_lsh_invalid_num_perm(self):
        """Test that invalid num_perm raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "num_perm": 0,
                    "seed": 42,
                    "threshold": 0.8,
                    "num_bands": 16,
                }
            )

    def test_constructor_lsh_invalid_seed(self):
        """Test that invalid seed raises ValueError for LSH."""
        with pytest.raises(ValueError):
            DeduplicationPromptAugmenter(
                algo_args={
                    "algorithm_name": "minhash_lsh",
                    "seed": -1,
                    "num_perm": 64,
                    "threshold": 0.8,
                    "num_bands": 16,
                }
            )

    def test_constructor_lsh_invalid_type(self):
        """Test that invalid type raises ValidationError for LSH."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeduplicationPromptAugmenter(
                algo_args={"algorithm_name": "minhash_lsh", "threshold": [0.8]}
            )

    def test_augment_lsh_empty_query(self, lsh_augmenter):
        """Test that LSH with empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = lsh_augmenter(message)
        assert result.query == ""

    def test_augment_lsh_exact_duplicates(self, lsh_augmenter):
        """Test LSH deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_lsh_multiple_exact_duplicates(self, lsh_augmenter):
        """Test LSH deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_lsh_no_duplicates(self, lsh_augmenter):
        """Test that LSH preserves text with no duplicates."""
        query = "The sky is blue. The grass is green. The sun is bright."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        # All sentences should be preserved
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_lsh_near_duplicates(self, lsh_low_threshold_augmenter):
        """Test LSH deduplication of near-duplicate sentences with low threshold."""
        query = (
            "The quick brown fox jumps. The quick brown fox jumped. Different topic."
        )
        message = AgentMessage(query=query)
        result = lsh_low_threshold_augmenter(message)
        # Near-duplicates should be removed with low threshold
        assert result.query != query

    def test_augment_lsh_unique_sentences_preserved(self, lsh_augmenter):
        """Test that LSH preserves all unique sentences."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    def test_augment_lsh_single_sentence(self, lsh_augmenter):
        """Test LSH with a single sentence."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert result.query == query

    def test_augment_lsh_single_word(self, lsh_augmenter):
        """Test LSH with a single word."""
        query = "Hello"
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert result.query == query

    def test_augment_lsh_multiple_exclamation(self, lsh_augmenter):
        """Test LSH with multiple exclamation marks."""
        query = "Wow! Wow! Amazing!"
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert result.query.count("Wow") == 1
        assert "Amazing" in result.query

    def test_augment_lsh_mixed_terminators(self, lsh_augmenter):
        """Test LSH with mixed sentence terminators."""
        query = "The sky is blue. The grass is green? The sun is bright!"
        message = AgentMessage(query=query)
        result = lsh_augmenter(message)
        assert "The sky is blue" in result.query
        assert "The grass is green" in result.query
        assert "The sun is bright" in result.query

    def test_augment_lsh_with_loop(self, lsh_augmenter):
        """Test LSH with loop configuration."""
        aug_with_loop = DeduplicationPromptAugmenter(
            algo_args={
                "algorithm_name": "minhash_lsh",
                "num_perm": 64,
                "seed": 42,
                "threshold": 0.8,
                "num_bands": 16,
            },
            loop=2,
        )
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = aug_with_loop(message)
        # The duplicate "Hello world." should be removed
        assert result.query.count("Hello world") == 1
        assert "This is a test" in result.query

    # --- Empty / None query ---

    def test_augment_empty_query(self, augmenter):
        """Test that empty query returns message unchanged."""
        message = AgentMessage(query="")
        result = augmenter(message)
        assert result.query == ""

    def test_augment_none_query(self, augmenter):
        """Test that None query returns message unchanged."""
        # AgentMessage requires a string query, so we test with empty string instead
        message = AgentMessage(query="")
        result = augmenter(message)
        assert result.query == ""

    # --- Exact duplicate sentences ---

    def test_augment_exact_duplicate_sentences(self, augmenter):
        """Test deduplication of exact duplicate sentences."""
        query = "Hello world. Hello world. This is a test."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # The duplicate "Hello world." should be removed
        assert result.query != query
        # Should still contain the unique parts
        assert "This is a test" in result.query

    def test_augment_multiple_exact_duplicates(self, augmenter):
        """Test deduplication of multiple exact duplicate sentences."""
        query = "Same sentence. Same sentence. Same sentence. Different one."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # "Same sentence." should appear only once
        assert result.query.count("Same sentence") == 1
        assert "Different one" in result.query

    def test_augment_no_duplicates(self, augmenter):
        """Test that text with no duplicates is preserved."""
        query = "First sentence. Second sentence. Third sentence."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # All sentences should be preserved
        assert "First sentence" in result.query
        assert "Second sentence" in result.query
        assert "Third sentence" in result.query

    # --- Near-duplicate sentences ---

    def test_augment_near_duplicate_sentences(self, low_threshold_augmenter):
        """Test deduplication of near-duplicate sentences with low threshold."""
        query = (
            "The quick brown fox jumps. The quick brown fox jumped. Different topic."
        )
        message = AgentMessage(query=query)
        result = low_threshold_augmenter(message)
        # Near-duplicates should be removed with low threshold
        assert result.query != query

    def test_augment_unique_sentences_preserved(self, augmenter):
        """Test that all unique sentences are preserved."""
        query = "Apple is a fruit. Carrot is a vegetable. Dog is an animal."
        message = AgentMessage(query=query)
        result = augmenter(message)
        assert "Apple is a fruit" in result.query
        assert "Carrot is a vegetable" in result.query
        assert "Dog is an animal" in result.query

    # --- Single sentence ---

    def test_augment_single_sentence(self, augmenter):
        """Test that a single sentence is preserved."""
        query = "This is a single sentence."
        message = AgentMessage(query=query)
        result = augmenter(message)
        assert result.query == query

    # --- Edge cases ---

    def test_augment_single_word_no_punctuation(self, augmenter):
        """Test text with no sentence boundaries."""
        query = "just one word"
        message = AgentMessage(query=query)
        result = augmenter(message)
        assert result.query == query

    def test_augment_multiple_exclamation_marks(self, augmenter):
        """Test deduplication with various sentence terminators."""
        query = "Wow! Wow! Amazing! Different."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # "Wow!" should appear only once
        assert result.query.count("Wow") == 1
        assert "Different" in result.query

    def test_augment_mixed_terminators(self, augmenter):
        """Test with mixed sentence terminators (.!?)."""
        query = "Hello. Wow! Really? Hello. Unique."
        message = AgentMessage(query=query)
        result = augmenter(message)
        # "Hello." should appear only once
        assert result.query.count("Hello") == 1
        assert "Unique" in result.query

    # --- Loop integration ---

    def test_augment_with_loop(self):
        """Test that the loop parameter works with DeduplicationPromptAugmenter."""
        aug = DeduplicationPromptAugmenter(
            algo_args={"num_perm": 64, "seed": 42, "threshold": 0.8}, loop=2
        )
        query = "Test. Test. Different."
        message = AgentMessage(query=query)
        result = aug(message)
        # After first pass, duplicates are removed; second pass should be a no-op
        assert result.query.count("Test") == 1
