import os

import filetype
from edgar import Company
from PIL import Image
from pydantic import BaseModel
from rich.progress import track


class EdgarDataLoader(BaseModel):
    company_name: str

    def load_data(
        self,
        form_type: str,
        upload_images: bool = True,
    ) -> list[dict[str, str]]:
        filings_10q = Company(self.company_name).get_filings(form=[form_type])
        filings_data = []
        for filing in track(
            filings_10q,
            description=f"Fetching {form_type} filings for {self.company_name}",
        ):

            current_filing_data = {
                "form_type": filing.primary_doc_description,
                "filing_date": filing.filing_date,
                "accession_no": filing.accession_no,
                "cik": filing.cik,
                "content": filing.markdown(),
            }

            attachment_texts = []
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
            current_filing_data["attachment_texts"] = attachment_texts

            filings_data.append(current_filing_data)

        return filings_data
