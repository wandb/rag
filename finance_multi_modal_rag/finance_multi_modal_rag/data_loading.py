import base64
import io
import os
import re

import filetype
from edgar import Company
from PIL import Image
from pydantic import BaseModel
from tqdm.auto import tqdm

from .llm_wrapper import MultiModalPredictor


def encode_image(image: Image.Image) -> str:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    encoded_string = f"data:image/png;base64,{encoded_string}"
    return str(encoded_string)


class EdgarDataLoader(BaseModel):
    company_name: str
    image_description_generator: MultiModalPredictor

    def generate_image_description(
        self, images: list[Image.Image], filing_summary: str
    ) -> list[str]:
        image_descriptions = []
        for image in tqdm(images, desc="Generating image descriptions:"):
            try:
                image_descriptions.append(
                    self.image_description_generator.predict(
                        user_prompts=[
                            f"""You are an expert finacial analyst tasked with genrating
    desciptions for images in financial filings given a summary of the financial filing
    and some important keywords present in the document along with the image.

    Here are some rules you should follow:
    1. If the image has text in it, you should first
    generate a description of the image and then extract the text in markdown format.
    2. If the image does not have text in it, you should generate a description of the image.
    3. You should frame your reply in markdown format.
    4. The description should be a list of bullet points under the markdown header "Description of the image".
    5. The extracted text should be under the markdown header "Extracted text from the image".
    6. If there are tables or tabular data in the image, you should extract the data in markdown format.
    7. You should pay attention to the financial filing and use the information to generate the description.

    Here is the financial filing's summary:

    ---
    {filing_summary}
    ---""",
                            encode_image(image),
                        ],
                    )
                )
            except Exception as e:
                print(e)
                image_descriptions.append("Error generating image description")
        return image_descriptions

    def summarize_filing(self, filing_data: str) -> list[str]:
        try:
            return self.image_description_generator.predict(
                user_prompts=[
                    f"""You are an expert finacial analyst tasked with genrating keywords for financial filings.
You should generate a summary of the financial filing and a list of important keywords from
the financial filing.

Here are some rules you should follow:
1. The summary should be a list of bullet points under the markdown header "Summary of the financial filing".
2. The keywords should be a list of keywords under the markdown header "Important keywords from the financial filing".

Here is the financial filing:

---
{filing_data}
---"""
                ]
            )
        except Exception as e:
            print(e)
            return ["Error generating summary"]

    def load_data(
        self,
        form_type: str,
        upload_images: bool = True,
        upload_image_descriptions: bool = True,
    ) -> list[dict[str, str]]:
        filings_10q = Company(self.company_name).get_filings(form=[form_type])
        filings_data = []
        for filing in tqdm(
            filings_10q,
            desc=f"Fetching {form_type} filings for {self.company_name}",
        ):

            filing_markdown = filing.markdown()
            filing_summary = self.summarize_filing(filing_markdown)
            current_filing_data = {
                "form_type": filing.primary_doc_description,
                "filing_date": re.findall(r"\((.*?)\)", filing.filing_date)[0].replace(
                    ", ", "-"
                ),
                "accession_no": filing.accession_no,
                "cik": filing.cik,
                "content": filing_markdown,
                "summary": filing_summary,
            }

            current_filing_data["images"] = []
            for idx, attachment in enumerate(filing.attachments):
                os.makedirs(
                    os.path.join("./attachments", self.company_name), exist_ok=True
                )
                attachment_path = os.path.join(
                    "./attachments",
                    self.company_name,
                    f"{idx}{attachment.extension}",
                )
                attachment.download(path=attachment_path)
                if upload_images and filetype.is_image(attachment_path):
                    image = Image.open(attachment_path)
                    try:
                        image.load()
                        current_filing_data["images"].append(image)
                    except Exception:
                        pass

            if upload_image_descriptions:
                current_filing_data["image_descriptions"] = (
                    self.generate_image_description(
                        images=current_filing_data["images"],
                        filing_summary=filing_summary,
                    )
                )

            filings_data.append(current_filing_data)

        return filings_data
