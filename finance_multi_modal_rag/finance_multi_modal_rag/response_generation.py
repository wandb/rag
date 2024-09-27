from typing import Any

import weave

from .llm_wrapper import MultiModalPredictor


class FinanceQABot(weave.Model):
    predictor: MultiModalPredictor
    retriever: weave.Model
    weave_corpus_dataset_address: str
    top_k: int
    _dataset: list[dict[str, Any]] = []

    def __init__(
        self,
        predictor: MultiModalPredictor,
        retriever: weave.Model,
        weave_corpus_dataset_address: str,
        top_k: int = 5,
    ):
        super().__init__(
            predictor=predictor,
            retriever=retriever,
            weave_corpus_dataset_address=weave_corpus_dataset_address,
            top_k=top_k,
        )
        self._dataset = [
            dict(row) for row in weave.ref(self.weave_corpus_dataset_address).get().rows
        ]

    def frame_user_prompt(self, query: str) -> str:
        retrieved_chunks = self.retriever.predict(query=query, top_k=self.top_k)
        user_prompt = ""
        for chunk in retrieved_chunks:
            date = chunk["metadata"]["filing_date"]
            accession_no = chunk["metadata"]["accession_no"]
            user_prompt += f"""# Report {accession_no} filed on {date}:

## An excerpt from the report

{chunk['retrieved_content']}\n\n
"""
            summary = self._dataset[chunk["metadata"]["document_idx"]]["summary"]
            user_prompt += f"""## Summary of the report
            
Here's a summary of the report along with the some important keywords and phrases present in the report:

{summary}\n\n
"""
            if chunk["metadata"]["number_of_images"] > 0:
                image_descriptions = "\n\n".join(
                    self._dataset[chunk["metadata"]["document_idx"]][
                        "image_descriptions"
                    ]
                )
                user_prompt += f"""## Image descriptions

{image_descriptions}\n\n
                """
        user_prompt += f"""You are suppossed to respond to the following query:

{query}
---
"""
        return user_prompt.strip()

    @weave.op()
    def predict(self, query: str):
        user_prompt = self.frame_user_prompt(query)
        return self.predictor.predict(
            system_prompt="""You are an expert and highly experienced financial analyst.
You are provided with the following excerpts from the companies 10-k filings for Tesla,
the electric car company. Your job is to answer the user's question is detail based on the
information provided.

Here are a few rules to follow:
1. You should pay close attention to the excerpts, especially the dates and other numbers in them.
2. You should pay attention to the image descriptions and use them to answer the user's question.
3. You should pay attention to the summary of the report and use it to answer the user's question.
4. You should pay attention to the keywords and phrases present in the report and use them to answer the user's question.
5. You should answer the user's question in detail and cite all the relevant information and references.
6. You should not make up any information. You should only use the information provided in the excerpts.
7. You should answer the user's question in a detailed and informative manner in English language.
            """,
            user_prompts=[user_prompt, query],
        )
