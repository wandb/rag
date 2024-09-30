import weave
from dotenv import load_dotenv

from finance_multi_modal_rag.retrieval import BGERetriever

load_dotenv()

weave.init(project_name="finance_multi_modal_rag")
dataset = weave.ref("TSLA_sec_filings:v8").get().rows
retriever = BGERetriever(
    model_name="BAAI/bge-small-en-v1.5",
    corpus=[str(item) for item in dataset[43]["image_descriptions"]],
)
retriever.create_index()
retriever.predict(query="what did elon say in the tweets that tesla reported?")
