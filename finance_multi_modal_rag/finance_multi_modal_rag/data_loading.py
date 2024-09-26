import base64
import io
import os

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
        self, images: list[Image.Image], filing_data: str
    ) -> list[str]:
        return [
            self.image_description_generator.predict(
                user_prompts=[
                    f"""You are an expert finacial analyst tasked with genrating
desciptions for images in financial filings. If the image has text in it, you should first
generate a description of the image and then extract the text in markdown format.

Here is the financial filing:

---
{filing_data}
---""",
                    encode_image(image),
                ],
            )
            for image in tqdm(images, desc="Generating image descriptions:")
        ]

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

            current_filing_data = {
                "form_type": filing.primary_doc_description,
                "filing_date": filing.filing_date,
                "accession_no": filing.accession_no,
                "cik": filing.cik,
                "content": filing.markdown(),
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
                        current_filing_data["images"], current_filing_data["content"]
                    )
                )

            filings_data.append(current_filing_data)

        return filings_data
