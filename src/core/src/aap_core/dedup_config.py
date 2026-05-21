from typing import Literal
from pydantic import BaseModel, Field


class BaseAlgoConfig(BaseModel):
    """Base configuration for deduplication algorithms."""

    algorithm_name: Literal["minhash", "minhash_lsh", "simhash"] = Field(
        ..., description="Name of the algorithm"
    )


class MinHashAlgoConfig(BaseAlgoConfig):
    """Configuration for C-MinHash based deduplication.

    Uses the **rensa** library for C-MinHash deduplication.
    """

    algorithm_name: Literal["minhash", "minhash_lsh", "simhash"] = "minhash"
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

    algorithm_name: Literal["minhash", "minhash_lsh", "simhash"] = "minhash_lsh"
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
