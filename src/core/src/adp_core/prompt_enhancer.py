import abc
import json
import os
from typing import List
from pydantic import Field, PrivateAttr
import tabulate
import pandas as pd

from .types import AgentMessage, BaseChain


class BasePromptEnhancer(BaseChain):
    """A base class to enhance / rewrite the prompt.

    There are two types of prompt enhancement:
    - Data enhancement: Give more context to the prompt by adding external data, either by using files (CSV, JSON, Markdown, etc.), database (SQL,...) or more advanced techniques like RAG, web search.
    - Structure enhancement: Rewrite / refine the prompt partially or entirely.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return super().__call__(message, **kwargs)


class IdentityPromptEnhancer(BasePromptEnhancer):
    """A prompt enhancer that does nothing.
    This serves as a default prompt enhancer."""

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return message


class DataFramePromptEnhancer(BasePromptEnhancer):
    """A prompt enhancer that uses a pandas dataframe to give more context to the prompt."""

    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {prompt} and {data}.
        Other parameters can be used and parsed""",
    )
    _data: str | None = PrivateAttr(None)

    def __init__(self, format: str = "{prompt}\n{data}", **kwargs):
        super().__init__()
        if "{prompt}" not in format or "{data}" not in format:
            raise ValueError("The format must contain at least {prompt} and {data}")
        self.format = format
        self._data = None

    def parse_pandas(
        self,
        data: pd.DataFrame,
        prettier: str | tabulate.TableFormat | None = None,
    ):
        """
        Parse a pandas dataframe into a string format.

        Args:
            data (pd.DataFrame): The pandas dataframe to be parsed.
            prettier (str | tabulate.TableFormat | None, optional): The prettier to be used. Defaults to None.
            The available table formats are from the [tabulate](https://pypi.org/project/tabulate/) library.

        Returns:
            self: The instance of the class.

        Notes:
            - If prettier is None, the dataframe is converted to a string using the to_string() method.
            - If prettier is not None, the dataframe is converted to a string using the tabulate module with the specified tablefmt.
        """
        if prettier is None:
            self._data = data.to_string()
        else:
            self._data = tabulate.tabulate(
                data.to_dict(), headers="keys", tablefmt=prettier
            )

        return self

    def parse_string(self, data: str):
        """
        Parse a string into a format that can be used by the prompt enhancer.

        Args:
            data (str): The string to be parsed.

        Returns:
            self: The instance of the class.
        """
        self._data = data
        return self

    def parse_dict(
        self, data: dict, prettier: str | tabulate.TableFormat | None = None
    ):
        """
        Parse a dictionary into a format that can be used by the prompt enhancer.

        Args:
            data (dict): The dictionary to be parsed.
            prettier (str | tabulate.TableFormat | None, optional): The prettier to be used. Defaults to None.
            The available table formats are from the [tabulate](https://pypi.org/project/tabulate/) library.

        Returns:
            self: The instance of the class.

        Notes:
            - If prettier is None, the dictionary is converted to a string using the str() method.
            - If prettier is not None, the dictionary is converted to a string using the tabulate module with the specified tablefmt.
        """
        if prettier is None:
            self._data = str(data)
        else:
            self._data = tabulate.tabulate(data, headers="keys", tablefmt=prettier)

        return self

    def parse_list(self, data: List[str], bullet_char: str = "-"):
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
        self.data = "\n".join([f"{bullet_char} {d}" for d in data])
        return self

    def parse_jsonl(self, path: str):
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
        self.parse_pandas(pd.DataFrame(line_dicts))
        return self

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if self._data is None:
            raise ValueError("Data not parsed yet")
        message.query = self.format.format(
            prompt=message.query, data=self._data, **kwargs
        )
        return message
