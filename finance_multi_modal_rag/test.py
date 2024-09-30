# import weave
# from dotenv import load_dotenv

# from finance_multi_modal_rag.retrieval import BGEImageRetriever

# load_dotenv()

# weave.init(project_name="finance_multi_modal_rag")
# dataset = weave.ref("TSLA_sec_filings:v8").get().rows
# retriever = BGEImageRetriever(model_name="BAAI/bge-small-en-v1.5")
# retriever.predict(
#     query="what did elon say in the tweets that tesla reported?",
#     image_descriptions=[str(item) for item in dataset[43]["image_descriptions"]],
# )


# import weave
# from dotenv import load_dotenv

# import wandb
# from finance_multi_modal_rag.retrieval import BGERetriever

# load_dotenv()

# weave.init(project_name="finance_multi_modal_rag")
# wandb.init(project="finance_multi_modal_rag", job_type="upload")
# retriever = BGERetriever(
#     weave_chunked_dataset_address="TSLA_sec_filings_chunks:v2",
#     model_name="BAAI/bge-small-en-v1.5",
# )
# retriever.create_index(index_persist_dir="./index", artifact_name="tsla-index")


import weave
from dotenv import load_dotenv

from finance_multi_modal_rag.llm_wrapper import MultiModalPredictor
from finance_multi_modal_rag.response_generation import FinanceQABot
from finance_multi_modal_rag.retrieval import BGEImageRetriever, BGERetriever

load_dotenv()

weave.init(project_name="finance_multi_modal_rag")
text_retriever = BGERetriever.from_wandb_artifact(
    artifact_address="geekyrakshit/finance_multi_modal_rag/tsla-index:latest",
    weave_chunked_dataset_address="TSLA_sec_filings_chunks:v1",
    model_name="BAAI/bge-small-en-v1.5",
)
image_retriever = BGEImageRetriever(model_name="BAAI/bge-small-en-v1.5")
finace_qa_bot = FinanceQABot(
    predictor=MultiModalPredictor(
        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        base_url="http://195.242.25.198:8032/v1",  # Replace with your base URL
    ),
    text_retriever=text_retriever,
    image_retriever=image_retriever,
    weave_corpus_dataset_address="TSLA_sec_filings:v8",
)
finace_qa_bot.predict(
    query="What are the major risks or considerations highlighted for investors?"
)
