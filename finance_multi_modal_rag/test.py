import weave
from dotenv import load_dotenv

from finance_multi_modal_rag.chunking import chunk_documents

load_dotenv()

weave.init(project_name="finance_multi_modal_rag")
chunk_documents(
    source_dataset_address="TSLA_sec_filings:v8",
    target_dataset_name="TSLA_sec_filings_chunks",
)