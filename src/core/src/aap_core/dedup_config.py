from typing import Literal
from pydantic import BaseModel, Field


class BaseAlgoConfig(BaseModel):
    """Base configuration for deduplication algorithms."""

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom"
    ] = Field(..., description="Name of the algorithm")


class MinHashAlgoConfig(BaseAlgoConfig):
    """Configuration for C-MinHash based deduplication.

    Uses the **rensa** library for C-MinHash deduplication.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom"
    ] = "minhash"
    threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0)",
    )
    num_perm: int = Field(
        default=128,
        gt=0,
        description="Number of permutations for MinHash",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class MinHashLSHAlgoConfig(BaseAlgoConfig):
    """Configuration for MinHash LSH based deduplication.

    Uses the **rensa** library for R-MinHash with Locality-Sensitive Hashing.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom"
    ] = "minhash_lsh"
    threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0)",
    )
    num_perm: int = Field(
        default=128,
        gt=0,
        description="Number of permutations for MinHash",
    )
    num_bands: int = Field(
        default=16,
        gt=0,
        description="Number of bands for LSH index. Higher = fewer false positives, more false negatives.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class BloomAlgoConfig(BaseAlgoConfig):
    """Configuration for Bloom Filter based deduplication.

    Uses the **rbloom** library for Bloom Filter based exact deduplication.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom"
    ] = "bloomfilter"
    expected_items: int = Field(
        default=10,
        gt=0,
        description="Expected number of items to add to the bloom filter",
    )
    false_positive_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Desired false positive rate (0.0 to 1.0)",
    )


class SimHashAlgoConfig(BaseAlgoConfig):
    """Configuration for SimHash based deduplication.

    Uses a custom numpy-based SimHash implementation for near-duplicate detection.
    SimHash produces a fingerprint for each document/sentence, and near-duplicates
    have similar fingerprints (low Hamming distance).
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom"
    ] = "simhash"
    hash_bits: Literal[64, 128] = Field(
        default=64,
        gt=0,
        description="Number of bits in the SimHash fingerprint",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0). "
        "Sentences with similarity >= threshold are considered duplicates.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )


class LSHBloomAlgoConfig(BaseAlgoConfig):
    """Configuration for LSH-Bloom Filter based deduplication.

    Uses MinHash with Locality-Sensitive Hashing (LSH) where each band's hash table
    is replaced by a Bloom filter for space-efficient near-duplicate detection.
    Inspired by the LSHBloom algorithm from https://arxiv.org/abs/2411.04257
    and the datasketch implementation.

    Each band's Bloom filter stores ``sum(hashvalues) % MersennePrime`` instead of
    all individual hash values, drastically reducing space usage for large-scale
    deduplication.
    """

    algorithm_name: Literal[
        "minhash", "minhash_lsh", "simhash", "bloomfilter", "lsh_bloom"
    ] = "lsh_bloom"
    threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection (0.0 to 1.0)",
    )
    num_perm: int = Field(
        default=128,
        gt=0,
        description="Number of permutations for MinHash",
    )
    num_bands: int = Field(
        default=16,
        gt=0,
        description="Number of bands for LSH index. Higher = fewer false positives, more false negatives.",
    )
    expected_items: int = Field(
        default=10,
        gt=0,
        description="Expected number of items to add to each Bloom filter",
    )
    false_positive_rate: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Desired false positive rate for each Bloom filter (0.0 to 1.0)",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
