# Multi-modal RAG for Finance

## Installation

Install dependencies using the following commands:

```bash
git clone https://github.com/soumik12345/physics-qa-bot
cd physics-qa-bot
pip install -U pip uv
uv sync
```

Next, you need to activate the virtual environment:

```bash
source .venv/bin/activate
```

Finally, you need to get a Cohere API key (depending on which model you use).

## Usage

First, you need to fetch the 10-Q filings from [Edgar database](https://www.sec.gov/edgar) and generate image descriptions using [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct).

```python
import weave
from edgar import set_identity

from finance_multi_modal_rag.data_loading import EdgarDataLoader
from finance_multi_modal_rag.llm_wrapper import MultiModalPredictor


def load_data(company_name: str, forms: list[str]):
    filings_data = []
    predictor = MultiModalPredictor(
        model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        base_url="http://195.242.25.198:8032/v1",
    )
    for form in forms:
        filings_data += EdgarDataLoader(
            company_name=company_name, image_description_generator=predictor
        ).load_data(form)
    weave.publish(weave.Dataset(name=f"{company_name}_sec_filings", rows=filings_data))
    return filings_data


if __name__ == "__main__":
    set_identity("<YOUR-NAME> <YOUR-EMAIL-ID>")
    weave.init(project_name="finance_multi_modal_rag")
    load_data("TSLA", ["10-Q"])
```

Next, we generate the chunks from our documents using the following code:

```python
import weave
from dotenv import load_dotenv

from finance_multi_modal_rag.chunking import chunk_documents

load_dotenv()

weave.init(project_name="finance_multi_modal_rag")
chunk_documents(
    source_dataset_address="TSLA_sec_filings:v6",
    target_dataset_name="TSLA_sec_filings_chunks",
)
```
