import os

from edgar import Company
from pydantic import BaseModel
from rich.progress import track


class EdgarDataLoader(BaseModel):
    company_name: str

    def load_data(
        self, form_type: str, download_attachments: bool = False
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

            if download_attachments:
                attachment_texts = []
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
                    if attachment.is_text():
                        with open(attachment_path, "r") as f:
                            attachment_texts.append(f.read())
                current_filing_data["attachment_texts"] = attachment_texts

            filings_data.append(current_filing_data)

        return filings_data
