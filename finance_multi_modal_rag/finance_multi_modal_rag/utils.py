import os
from functools import partial

import cohere
import tiktoken


def tokenize_text(text: str, model: str = "command-r") -> list[str]:
    """
    Tokenizes the given text using the specified model.

    Args:
        text (str): The text to be tokenized.
        model (str): The model to use for tokenization. Defaults to "command-r".

    Returns:
        List[str]: A list of tokens.
    """
    co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
    return co.tokenize(text=text, model=model, offline=True)


def cohere_length_function(text, model="command-r"):
    """
    Calculate the length of the tokenized text using the specified model.

    Args:
        text (str): The text to be tokenized and measured.
        model (str): The model to use for tokenization. Defaults to "command-r".

    Returns:
        int: The number of tokens in the tokenized text.
    """
    return len(tokenize_text(text, model=model).tokens)


def tiktoken_length_function(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


length_function_command_r = partial(cohere_length_function, model="command-r")
length_function_command_r_plus = partial(cohere_length_function, model="command-r-plus")
length_function_cl100k_base = partial(
    tiktoken_length_function, encoding_name="cl100k_base"
)
