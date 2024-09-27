import weave
from dotenv import load_dotenv

from finance_multi_modal_rag.llm_wrapper import MultiModalPredictor
from finance_multi_modal_rag.response_generation import FinanceQABot
from finance_multi_modal_rag.retrieval import BGERetriever

load_dotenv()

weave.init(project_name="finance_multi_modal_rag")
retriever = BGERetriever.from_wandb_artifact(
    artifact_address="geekyrakshit/finance_multi_modal_rag/tsla-index:latest",
    weave_chunked_dataset_address="TSLA_sec_filings_chunks:v1",
    model_name="BAAI/bge-small-en-v1.5",
)
finace_qa_bot = FinanceQABot(
    predictor=MultiModalPredictor(
        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        base_url="http://195.242.25.198:8032/v1",
    ),
    retriever=retriever,
    weave_corpus_dataset_address="TSLA_sec_filings:v8",
)
finace_qa_bot.predict(query="what did elon say in the tweets that tesla reported?")
