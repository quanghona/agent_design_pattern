# Retriever

## Overview

The `Retriever` component provides a unified interface for injecting external data into an agent's message context. Retrievers are responsible for fetching data from any source — databases, files, APIs, or in-memory structures — and attaching it to the `AgentMessage.context` dictionary so that downstream agents and prompt augmenters can consume it. This abstraction enables RAG pipelines, structured data retrieval, and arbitrary data injection to be composed seamlessly with agents and chains.

## Architecture

Retrievers sit between data sources and agents in the component hierarchy. A retriever receives an `AgentMessage`, retrieves data from its source, and returns the same message with the retrieved data stored in `message.context` under a configurable key. The retriever can optionally chain a `post_process` step (e.g., a reranker or summarizer) after retrieval.

Retrievers integrate with the `BaseChain` interface, making them directly composable with other components in chains and orchestration patterns. The `BaseRetriever` abstract base class defines the contract; `DataFrameRetriever` is the primary concrete implementation, supporting data from pandas DataFrames, strings, dictionaries, iterables, and JSONL files.

![diagram](img/retriever-architecture.jpg)

## Key Concepts

### BaseRetriever: The Retrieval Contract

`BaseRetriever` is an abstract base class that inherits from `BaseChain`. It defines two key members:

- **`retrieve(message, **kwargs)`**: An abstract method that subclasses implement to fetch data from a specific source and return a (possibly modified) `AgentMessage`.
- **`post_process`**: An optional `BaseChain` that runs after `retrieve()`. This enables post-processing steps such as reranking, summarization, or filtering without cluttering the retriever implementation.

The `__call__` method orchestrates the pipeline: it first calls `retrieve()`, then applies `post_process` if configured, and returns the final `AgentMessage`.

### DataFrameRetriever: In-Memory Data Retrieval

`DataFrameRetriever` is the primary concrete retriever. It stores pre-formatted data as a string internally and injects it into the message context on retrieval. The class is designed around factory methods that accept data in various formats and convert it to a string representation.

The retriever uses a `data_key` field (default: `"context.data"`) to determine where the retrieved data is stored in the message context. The key must start with the `"context."` prefix, which is enforced by a Pydantic validator.

### Data Formats and Factory Methods

`DataFrameRetriever` provides five factory class methods, each accepting a different input format:

| Factory Method | Input Type | Description |
|---|---|---|
| `from_pandas` | `pd.DataFrame` | Converts a pandas DataFrame to a string using `to_string()` or `tabulate` with a specified table format. |
| `from_string` | `str` | Wraps a raw string directly as the retrievable data. |
| `from_dict` | `dict` | Converts a dictionary to a string, optionally using `tabulate` for tabular data. |
| `from_iterable` | `Iterable[str]` | Joins a list of strings with bullet characters for list-style data. |
| `from_jsonl` | `str` (file path) | Reads a JSONL file, converts each line to a dictionary, builds a pandas DataFrame, and delegates to `from_pandas`. |

### Formatting Options

When using `from_pandas` or `from_dict`, you can control the output format via the `prettier` parameter:

- **`None`** (default): Uses `pandas.DataFrame.to_string()` for pandas input, or Python's built-in `str()` for dictionaries.
- **A `tabulate.TableFormat` or format name string**: Uses the `tabulate` library to render the data as a formatted table. Supported formats include all formats from the [tabulate](https://pypi.org/project/tabulate/) library (e.g., `"grid"`, `"pipe"`, `"github"`) as well as the [toon](https://github.com/toon-format/spec) format (currently in beta).
- **Additional `**kwargs`**: Passed through to the underlying formatting function (`to_string()` or `tabulate()`), allowing fine-grained control over headers, index display, and other formatting options.

### Context Key Management

The `data_key` field controls where retrieved data is stored in the message context. The key is split on `"context."` to extract the context dictionary key:

```python
# data_key = "context.data" → stored at message.context["data"]
# data_key = "context.retrieved_docs" → stored at message.context["retrieved_docs"]
```

This design allows multiple retrievers to store data under different keys in the same message context, enabling complex multi-source retrieval scenarios.

## Usage

### Basic Example: Retrieving from a pandas DataFrame

```python
import pandas as pd
from aap_core import AgentMessage
from aap_core.retriever import DataFrameRetriever

# Create sample data
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [30, 25, 35],
    "City": ["New York", "London", "Tokyo"],
})

# Create a retriever from the DataFrame
retriever = DataFrameRetriever.from_pandas(df, prettier="grid")

# Use the retriever on a message
message = AgentMessage(query="Tell me about the people")
result = retriever(message)

print(result.context["data"])
# +---------+-----+----------+
# | Name    |   Age | City     |
# |---------+-------+----------|
# | Alice   |    30 | New York |
# | Bob     |    25 | London   |
# | Charlie |    35 | Tokyo    |
```

### Basic Example: Retrieving from a JSONL File

```python
from aap_core import AgentMessage
from aap_core.retriever import DataFrameRetriever

# Create a retriever from a JSONL file
retriever = DataFrameRetriever.from_jsonl(
    "data/products.jsonl",
    prettier="pipe",
    headers="keys",
    index=False,
)

# Use the retriever on a message
message = AgentMessage(query="What products do you have?")
result = retriever(message)

print(result.context["data"])
```

### Advanced Example: Retrieval with Post-Processing

```python
from aap_core import AgentMessage, BaseChain
from aap_core.retriever import DataFrameRetriever


class SummarizingPostProcess(BaseChain):
    """A simple post-processor that summarizes retrieved data."""

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # In practice, this would call an LLM to summarize the data
        data = message.context.get("data", "")
        message.context["summary"] = f"[Summarized: {len(data)} characters of data retrieved]"
        return message


# Create a retriever with post-processing
retriever = DataFrameRetriever.from_pandas(
    pd.DataFrame({"product": ["A", "B", "C"], "price": [10, 20, 30]}),
    prettier="grid",
)
retriever.post_process = SummarizingPostProcess()

message = AgentMessage(query="Show me product info")
result = retriever(message)

# Both raw data and summary are available
print(result.context["data"])     # The formatted table
print(result.context["summary"])  # The post-processed summary
```

### Advanced Example: Custom Data Key for Multi-Source Retrieval

```python
from aap_core import AgentMessage
from aap_core.retriever import DataFrameRetriever

# Two retrievers storing data under different keys
products_retriever = DataFrameRetriever.from_pandas(
    pd.DataFrame({"product": ["A", "B"], "price": [10, 20]}),
    data_key="context.products",
    prettier="grid",
)

reviews_retriever = DataFrameRetriever.from_pandas(
    pd.DataFrame({"product": ["A", "B"], "rating": [4.5, 3.8]}),
    data_key="context.reviews",
    prettier="grid",
)

# Chain both retrievers
message = AgentMessage(query="Compare products")
message = products_retriever(message)
message = reviews_retriever(message)

# Both datasets are available in the context
print(message.context["products"])  # Product table
print(message.context["reviews"])   # Reviews table
```

## Configuration

### BaseRetriever Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `post_process` | `BaseChain \| None` | `None` | An optional post-processing chain that runs after `retrieve()`. Useful for reranking, summarization, or filtering retrieved data. |

### DataFrameRetriever Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_key` | `str` | `"context.data"` | The key under which retrieved data is stored in `message.context`. Must start with `"context."`. |

### Factory Method Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `pd.DataFrame \| str \| dict \| Iterable[str]` | *required* | The input data to retrieve. Type depends on the factory method used. |
| `data_key` | `str` | `"context.data"` | The context key for storing retrieved data. |
| `prettier` | `str \| tabulate.TableFormat \| None` | `None` | The table format for `from_pandas` and `from_dict`. Use `None` for plain string output, or a tabulate format name (e.g., `"grid"`, `"pipe"`) for formatted tables. |
| `bullet_char` | `str` | `"-"` | The bullet character for `from_iterable`. |
| `**kwargs` | *varies* | — | Additional arguments passed to the underlying formatting function (`to_string()`, `tabulate()`, etc.). |

## API Reference

See the full API reference: [`BaseRetriever`][aap_core.retriever.BaseRetriever] and [`DataFrameRetriever`][aap_core.retriever.DataFrameRetriever]

::: aap_core.retriever.BaseRetriever
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.retriever.DataFrameRetriever
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Agent](agent.md) — How agents consume retrieved data from `message.context`
- [Prompt Augmenter](prompt_augmenter.md) — How retrieved data is injected into prompts
- [Chain](chain.md) — How retrievers compose with chains in multi-step workflows
- [pandas](https://pandas.pydata.org/docs/) — The pandas library for DataFrame creation and manipulation
- [tabulate](https://pypi.org/project/tabulate/) — The tabulate library for table formatting options
