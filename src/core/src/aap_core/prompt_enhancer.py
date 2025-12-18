import abc
import json
import os
from collections.abc import Sequence
from typing import List, Tuple

import pandas as pd
import tabulate
from pydantic import Field, PrivateAttr, field_validator

# import toon_format
from .types import AgentMessage, BaseChain, ContentType


class BasePromptEnhancer(BaseChain):
    """A base class to enhance / rewrite the prompt.

    There are two types of prompt enhancement:
    - Data enhancement: Give more context to the prompt by adding external data, either by using files (CSV, JSON, Markdown, etc.), database (SQL,...) or more advanced techniques like RAG, web search.
    - Structure enhancement: Rewrite / refine the prompt partially or entirely.
    """

    @abc.abstractmethod
    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return super().__call__(message, **kwargs)


class IdentityPromptEnhancer(BasePromptEnhancer):
    """A prompt enhancer that does nothing.
    This serves as a default prompt enhancer."""

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return message


class DataFramePromptEnhancer(BasePromptEnhancer):
    """A prompt enhancer that uses a pandas dataframe to give more context to the prompt."""

    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {query} and {data}.
        Other parameters can be used and parsed""",
    )
    _data: str | None = PrivateAttr(None)

    @field_validator("format")
    @classmethod
    def check_starts_with_prompt_and_data(cls, v: str) -> str:
        if "{query}" not in v or "{data}" not in v:
            raise ValueError("The format must contain at least {query} and {data}")
        return v

    @classmethod
    def from_pandas(
        cls,
        format: str,
        data: pd.DataFrame,
        prettier: str | tabulate.TableFormat | None = None,
    ):
        """
        Parse a pandas dataframe into a string format.

        Args:
            data (pd.DataFrame): The pandas dataframe to be parsed.
            prettier (str | tabulate.TableFormat | None, optional): The prettier to be used. Defaults to None.
            The available table formats are from the [tabulate](https://pypi.org/project/tabulate/) library.
            [toon](https://github.com/toon-format/spec) are also supported

        Returns:
            self: The instance of the class.

        Notes:
            - If prettier is 'toon', the dataframe is converted to a string following toon specification
            - If prettier is None, the dataframe is converted to a string using the to_string() method.
            - If prettier is not None, the dataframe is converted to a string using the tabulate module with the specified tablefmt.
        """
        enhancer = DataFramePromptEnhancer(format=format)
        if prettier is None:
            enhancer._data = data.to_string()
        # elif prettier == "toon":      # toon_format currently is beta release, waiting 1.0.0 release
        #     enhancer._data = toon_format.encode(data.to_dict())
        else:
            enhancer._data = tabulate.tabulate(
                data.to_dict(), headers="keys", tablefmt=prettier
            )

        return enhancer

    @classmethod
    def from_string(cls, format: str, data: str):
        """
        Parse a string into a format that can be used by the prompt enhancer.

        Args:
            data (str): The string to be parsed.

        Returns:
            self: The instance of the class.
        """
        enhancer = DataFramePromptEnhancer(format=format)
        enhancer._data = data
        return enhancer

    @classmethod
    def from_dict(
        cls, format: str, data: dict, prettier: str | tabulate.TableFormat | None = None
    ):
        """
        Parse a dictionary into a format that can be used by the prompt enhancer.

        Args:
            data (dict): The dictionary to be parsed.
            prettier (str | tabulate.TableFormat | None, optional): The prettier to be used. Defaults to None.
            The available table formats are from the [tabulate](https://pypi.org/project/tabulate/) library.
            [toon](https://github.com/toon-format/spec) are also supported

        Returns:
            self: The instance of the class.

        Notes:
            - If prettier is 'toon', the dataframe is converted to a string following toon specification
            - If prettier is None, the dictionary is converted to a string using the str() method.
            - If prettier is not None, the dictionary is converted to a string using the tabulate module with the specified tablefmt.
        """
        enhancer = DataFramePromptEnhancer(format=format)
        if prettier is None:
            enhancer._data = str(data)
        # elif prettier == "toon":      # toon_format currently is beta release, waiting 1.0.0 release
        #     enhancer._data = toon_format.encode(data.to_dict())
        else:
            enhancer._data = tabulate.tabulate(data, headers="keys", tablefmt=prettier)

        return enhancer

    @classmethod
    def from_list(cls, format: str, data: List[str], bullet_char: str = "-"):
        """
        Parse a list of strings into a format that can be used by the prompt enhancer.

        Args:
            data (List[str]): The list of strings to be parsed.
            bullet_char (str, optional): The character to be used as the bullet. Defaults to "-".

        Returns:
            self: The instance of the class.

        Notes:
            - The list of strings is converted to a string using the join() method with the bullet character as the separator.
        """
        enhancer = DataFramePromptEnhancer(format=format)
        enhancer._data = "\n".join([f"{bullet_char} {d}" for d in data])
        return enhancer

    @classmethod
    def from_jsonl(cls, format: str, path: str | os.PathLike[str]):
        """
        Parse a JSONL file into a format that can be used by the prompt enhancer.

        Args:
            path (str): The path to the JSONL file.

        Returns:
            self: The instance of the class.

        Raises:
            FileNotFoundError: If the file specified by path does not exist.

        Notes:
            - The JSONL file is parsed line by line using the json.loads() method.
            - The parsed data is converted to a pandas DataFrame and passed to the parse_pandas() method.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")

        # Implementation provided by https://stackoverflow.com/a/74406107
        with open(path, "r") as f:
            lines = f.read().splitlines()

        line_dicts = [json.loads(line) for line in lines]
        return DataFramePromptEnhancer.from_pandas(format, pd.DataFrame(line_dicts))

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if self._data is None:
            raise ValueError("Data not parsed yet")
        message.query = self.format.format(
            query=message.query, data=self._data, **kwargs
        )
        return message


class BaseRAGPromptEnhancer(BasePromptEnhancer):
    @abc.abstractmethod
    def _search(self, query: str, **kwargs) -> Sequence[Tuple[ContentType, str]]:
        """"""
        pass

    @abc.abstractmethod
    def _format(
        self, message: AgentMessage, data: Sequence[Tuple[ContentType, str]]
    ) -> AgentMessage:
        # For string type, the field we need to touch is `query`. For other types, the media `query_media` and `query_media_type` needed to modify
        pass

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        data = self._search(message.query, **kwargs)
        prompt = self._format(message, data)
        return prompt
