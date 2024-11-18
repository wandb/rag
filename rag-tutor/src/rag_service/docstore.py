import asyncio
from typing import Any, Sequence

import lancedb
import litellm
import weave
from litellm import arerank
from litellm.caching import Cache, LiteLLMCacheType

from src.rag_service.query_expansion import query_expansion
from src.rag_service.retrieval.models import DocumentChunk, RetrievalResults
from src.rag_service.utils import CACHE_DIR, dedupe_docs, embed_documents

disk_cache_dir = CACHE_DIR / "litellm"
litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(disk_cache_dir))
litellm.suppress_debug_info = True


def iter_data(
    corpus: list[DocumentChunk],
    embed_field="embed_content",
    embed_model="text-embedding-3-small",
    embed_dims=512,
    batch_size=500,
):
    dict_corpus = [doc.model_dump(mode="json", exclude={"links"}) for doc in corpus]
    docs = [doc[embed_field] for doc in dict_corpus]
    doc_batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
    corpus_batches = [
        dict_corpus[i : i + batch_size] for i in range(0, len(dict_corpus), batch_size)
    ]
    output = []
    for doc_batch, corpus_batch in zip(doc_batches, corpus_batches):
        vectors = embed_documents(doc_batch, embed_model, embed_dims)
        for doc, vector in zip(corpus_batch, vectors):
            doc["vector"] = vector
            output.append(doc)
        yield output


class HybridRetriever(weave.Model):
    uri: str = "data/documents.db"
    table_name: str = "documents"
    embed_field: str = "embed_content"
    fts_field: str = "embed_content"
    rerank_model: str = "cohere/rerank-english-v3.0"
    use_query_expansion: bool = False
    db: Any = None
    table: Any = None

    def index(self, corpus: list[DocumentChunk]):
        self.db = lancedb.connect(self.uri)
        self.table = self.db.create_table(
            self.table_name,
            mode="overwrite",
            exist_ok=True,
            data=iter_data(corpus, embed_field=self.embed_field),
        )
        self.table.create_fts_index(self.fts_field, replace=True)
        return True

    @weave.op
    async def retrieve(self, query, k=2):
        query_vector = embed_documents([query])
        query_vector = query_vector[0]
        if self.table is None:
            self.table = self.db.open_table(self.table_name)
        result = (
            self.table.search(
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns=[self.fts_field],
            )
            .vector(query_vector)
            .text(query)
            .limit(k)
            .to_list()
        )
        for item in result:
            item.pop("vector")
            item.pop("_relevance_score")
        return result

    @weave.op
    async def rerank(self, query, docs: list[dict], top_n=None):
        """
        Reranks the given documents based on their relevance to the query.

        Args:
            query (str): The query string.
            docs (List[Dict[str, Any]]): A list of documents to be reranked.
            top_n (int, optional): The number of top documents to return. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of reranked documents with relevance scores.
        """

        documents = [doc[self.embed_field] for doc in docs]
        response = await arerank(
            model=self.rerank_model,
            custom_llm_provider="cohere",
            query=query,
            documents=documents,
            top_n=top_n or len(docs),
        )

        outputs = []
        for doc in response.results:
            reranked_doc = docs[doc["index"]]
            outputs.append(reranked_doc)
        return outputs[:top_n]

    @weave.op
    async def retrieve_and_rerank(self, query, k=15) -> Sequence[DocumentChunk]:
        documents = await self.retrieve(query, k=k * 20)
        deduped_docs = dedupe_docs(documents, key="content")
        reranked_docs = await self.rerank(query, deduped_docs, top_n=k * 2)
        return reranked_docs

    @weave.op
    async def invoke(self, query, limit=5) -> Sequence[RetrievalResults]:
        if self.use_query_expansion:
            queries = await query_expansion(query)
            tasks = [self.retrieve_and_rerank(q, k=limit) for q in [query] + queries]
            all_docs = await asyncio.gather(*tasks)
            final_results = [
                [DocumentChunk(**doc) for doc in results] for results in all_docs
            ]
            return [
                RetrievalResults(query=query, results=results)
                for results in final_results
            ]
        else:
            results = await self.retrieve_and_rerank(query, k=limit)
            final_results = [DocumentChunk(**doc) for doc in results][:limit]
            return [RetrievalResults(query=query, results=final_results)]

    @classmethod
    def load(cls, uri="data/documents.db", table_name="documents"):
        instance = cls()
        instance.uri = uri
        instance.db = lancedb.connect(instance.uri)
        instance.table_name = table_name
        instance.table = instance.db.open_table(instance.table_name)
        return instance


async def main():
    #

    # import json
    #
    # files = [
    #     "data/cookbook_docs.jsonl",
    #     "data/blog_articles.jsonl",
    #     "data/arxiv_articles.jsonl",
    # ]
    # corpus = []
    # for file in files:
    #     with open(file, "r") as f:
    #         for line in f:
    #             doc = json.loads(line)
    #             doc = Document(**doc)
    #             corpus.append(doc)
    # logger.info(f"loaded: {len(corpus)} documents")
    #
    # from joblib import Parallel, delayed
    #
    # corpus_chunks = Parallel(n_jobs=-1)(
    #     delayed(chunk_doc)(doc, length_fn=length_fn) for doc in corpus
    # )
    #
    # corpus_chunks = flatten_sequence(corpus_chunks)
    # logger.info(f"chunked: {len(corpus_chunks)} documents")
    retriever = HybridRetriever.load()
    # retriever.index(corpus_chunks)
    from time import perf_counter

    start = perf_counter()
    results = await retriever.invoke("What is Contextual Retrieval?", 20)
    end = perf_counter()
    print(len(results))
    for doc in results:
        print(doc.as_str)
        print("\n\n")
        print("-" * 100)
        print("\n\n")
    print(f"Time taken: {end - start:.2f} seconds")

    # data_iter = iter(iter_data(corpus_chunks, batch_size=2))
    # sample = next(data_iter)
    # pprint(sample)
    #


if __name__ == "__main__":

    asyncio.run(main())
