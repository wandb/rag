"""
This module contains functions and classes for splitting text into chunks based on sentence boundaries and semantic similarity.

Source: https://github.com/wandb/edu/blob/main/rag-advanced/notebooks/scripts/chunking.py
"""

import re
from functools import partial
from typing import Callable, List, Optional

import numpy as np
import weave
from blingfire import text_to_sentences
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances
from tqdm.notebook import tqdm

from .embedding import sync_embed
from .utils import length_function

CHUNK_SIZE = 512


def sentence_splitter(text: str) -> List[str]:
    """
    Splits the input text into sentences.

    Args:
        text (str): The input text to be split into sentences.

    Returns:
        List[str]: A list of sentences.
    """
    return text_to_sentences(text).split("\n")


def split_into_chunks(
    text: str, length_function: Callable[[str], int], max_tokens: int
) -> List[str]:
    """
    Splits the input text into chunks based on the specified maximum number of tokens.

    Args:
        text (str): The input text to be split into chunks.
        length_function (Callable[[str], int]): A function that calculates the number of tokens in a given string.
        max_tokens (int): The maximum number of tokens allowed in each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    sentences = sentence_splitter(text)

    n_tokens = [length_function("\n" + sentence) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append("\n".join(chunk))
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    if chunk:
        chunks.append("\n".join(chunk))

    return chunks


class KamradtModifiedChunker:
    """
    A chunker that splits text into chunks of approximately a specified average size based on semantic similarity.

    This implementation is adapted from
    https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/kamradt_modified_chunker.py
    """

    def __init__(
        self,
        avg_chunk_size: int = CHUNK_SIZE,
        min_chunk_size: int = 50,
        embedding_function: Optional[Callable] = None,
        length_function: Optional[Callable] = None,
    ):
        """
        Initializes the KamradtModifiedChunker.

        Args:
            avg_chunk_size (int): The average size of each chunk.
            min_chunk_size (int): The minimum size of each chunk.
            embedding_function (Optional[Callable]): A function to compute embeddings for sentences.
            length_function (Optional[Callable]): A function to compute the length of a sentence.
        """
        self.splitter = partial(
            split_into_chunks,
            length_function=length_function,
            max_tokens=min_chunk_size,
        )

        self.avg_chunk_size = avg_chunk_size
        if embedding_function is None:
            embedding_function = sync_embed
        self.embedding_function = embedding_function
        self.min_chunk_size = min_chunk_size
        if length_function is None:
            length_function = len
        self.length_function = length_function

    def combine_sentences(self, sentences, buffer_size=1, sep="\n"):
        """
        Combines sentences with a buffer size.

        Args:
            sentences (list): List of sentences to combine.
            buffer_size (int): The buffer size for combining sentences.
            sep (str): The separator to use when combining sentences.

        Returns:
            list: List of sentences with combined sentences.
        """
        n = len(sentences)
        for i in range(n):
            start = max(0, i - buffer_size)
            end = min(n, i + buffer_size + 1)

            combined = sep.join(sentences[j]["sentence"] for j in range(start, end))

            sentences[i]["combined_sentence"] = combined

        return sentences

    def calculate_cosine_distances(self, sentences):
        """
        Calculates cosine distances between combined sentences.

        Args:
            sentences (list): List of sentences with combined sentences.

        Returns:
            tuple: A tuple containing a list of distances and the updated sentences.
        """
        if len(sentences) <= 1:
            return [], sentences

        combined_sentences = [sentence["combined_sentence"] for sentence in sentences]
        embeddings = self.embedding_function(combined_sentences)
        embedding_matrix = np.array(embeddings)

        if embedding_matrix.shape[0] <= 1:
            return [], sentences

        distances = cosine_distances(
            embedding_matrix[:-1], embedding_matrix[1:]
        ).diagonal()

        for i, distance in enumerate(distances):
            sentences[i]["distance_to_next"] = distance

        return distances.tolist(), sentences

    def split_text(self, text):
        """
        Splits the input text into chunks of approximately the specified average size based on semantic similarity.

        Args:
            text (str): The input text to be split into chunks.

        Returns:
            list of str: The list of text chunks.
        """
        sentences_strips = self.splitter(text)
        if len(sentences_strips) <= 1:
            # Return the original text as a single chunk
            return [text]

        sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(sentences_strips)
        ]

        sentences = self.combine_sentences(sentences, 3)

        distances, sentences = self.calculate_cosine_distances(sentences)

        total_tokens = sum(
            self.length_function(sentence["sentence"]) for sentence in sentences
        )
        avg_chunk_size = self.avg_chunk_size
        number_of_cuts = total_tokens // avg_chunk_size

        lower_limit = 0.0
        upper_limit = 1.0

        distances_np = np.array(distances)

        while upper_limit - lower_limit > 1e-6:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)

            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold

        indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold]

        start_index = 0

        chunks = []

        for index in indices_above_thresh:
            end_index = index

            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            start_index = index + 1

        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks


@weave.op()
def chunk_single_document(
    document: dict[str, str | list[str, Image.Image]],
    document_idx: int,
    chunk_size: int = CHUNK_SIZE,
    cohere_model: str = "command-r",
):
    chunker = KamradtModifiedChunker(
        avg_chunk_size=chunk_size,
        length_function=partial(length_function, model=cohere_model),
    )
    filing_date = re.findall(r"\((.*?)\)", document["filing_date"])[0].replace(
        ", ", "-"
    )
    text = "# Report filed at" + filing_date + "\n\n" + str(document["content"])
    text += "\n\n# Report Summary and Keywords\n\n" + str(document["summary"])
    text += "\n\n# Image Descriptions\n" + str(document["image_descriptions"])
    chunks = chunker.split_text(text)
    return [
        {
            "cleaned_content": chunk,
            "metadata": {
                "document_idx": document_idx,
                "filing_date": filing_date,
                "accession_no": document["accession_no"],
                "number_of_images": len(document["images"]),
                "parsed_tokens": length_function(chunk),
            },
        }
        for chunk in chunks
    ]


@weave.op()
def chunk_documents(
    source_dataset_address: str,
    chunk_size: int = CHUNK_SIZE,
    cohere_model: str = "command-r",
    target_dataset_name: Optional[str] = None,
):
    dataset_rows = [dict(row) for row in weave.ref(source_dataset_address).get().rows]
    document_chunks = []
    for row in tqdm(dataset_rows, desc="Chunking documents:"):
        document_chunks += chunk_single_document(
            row, dataset_rows.index(row), chunk_size, cohere_model
        )
    if target_dataset_name is not None:
        weave.publish(weave.Dataset(name=target_dataset_name, rows=document_chunks))
    return document_chunks
