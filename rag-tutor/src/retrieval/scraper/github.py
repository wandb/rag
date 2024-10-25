import fnmatch
import io
import tempfile
import warnings
import zipfile
from pathlib import Path

import httpx
from models import Document
from nbconvert import MarkdownExporter
from nbformat import reads
from nbformat.validator import normalize as nb_normalize
from traitlets.config import Config
from utils import cleanup_text, html_to_md, length_fn, load_html_content, md_to_html

warnings.filterwarnings("ignore")


FILE_TYPES = ["*.py", "*.md", "*.ipynb"]
BLACKLIST = [".vscode", ".github", ".devcontainer", "node_modules"]

SOURCE_TYPE_MAP = {".ipynb": "Notebook", ".md": "Webpage", ".py": "Source Code"}


def read_notebook(content):
    notebook = reads(content, as_version=4)
    _, notebook = nb_normalize(notebook, version=4, strip_invalid_metadata=True)
    conf = Config()
    conf.MarkdownExporter.preprocessors = [
        "nbconvert.preprocessors.ClearOutputPreprocessor"
    ]
    md_exporter = MarkdownExporter(config=conf, template="classic")
    body, _ = md_exporter.from_notebook_node(notebook)
    return body


def should_include_file(
    file_path: Path, gitignore_patterns: list[str], blacklist: list[str]
) -> bool:
    """
    Check if a file should be included based on .gitignore patterns and blacklist.

    :param file_path: Path object of the file to check
    :param gitignore_patterns: List of .gitignore patterns
    :param blacklist: List of blacklisted directory or file names
    :return: True if the file should be included, False otherwise
    """
    # Check against blacklist
    for item in blacklist:
        if item in file_path.parts:
            return False

    # Check against .gitignore patterns
    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(str(file_path), pattern):
            return False

    return True


class RepoScrapper:
    def __init__(
        self,
        owner: str,
        repo: str,
        content_roots: list[str] = None,
        file_types: list = None,
    ):
        self.owner = owner
        self.repo = repo
        self.repo_url = None
        self.tag = None
        self.branch = None
        self.content_roots = [""] if content_roots is None else content_roots
        self.file_types = file_types if file_types else FILE_TYPES

    async def __aenter__(self):
        self.set_repo_url()
        self.set_tag_or_branch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_repo_url(self):
        self.repo_url = f"https://github.com/{self.owner}/{self.repo}"

    def set_tag_or_branch(self):
        response = httpx.get(
            f"https://api.github.com/repos/{self.owner}/{self.repo}/releases/latest"
        )
        if response.status_code == 200:
            tags = response.json()
            self.tag = tags["tag_name"]
        else:
            response = httpx.get(
                f"https://api.github.com/repos/{self.owner}/{self.repo}"
            )
            if response.status_code == 200:
                branch = response.json()["default_branch"]
                self.branch = branch

    def get_zip_url(self):
        if self.tag:
            return f"{self.repo_url}/archive/refs/tags/{self.tag}.zip"
        else:
            return f"{self.repo_url}/archive/refs/heads/{self.branch}.zip"

    def get_content_root(self, root):
        if self.tag:
            return f"{self.repo}-{self.tag.replace('v', '')}/{root}"
        else:
            return f"{self.repo}-{self.branch}/{root}"

    async def get_zip_data(self):
        url = self.get_zip_url()
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return io.BytesIO(response.content)

    async def extract_zip_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_data = await self.get_zip_data()
            with zipfile.ZipFile(zip_data) as zip_file:
                zip_file.extractall(temp_dir)

                # Read .gitignore file if it exists
                gitignore_path = (
                    Path(temp_dir) / self.get_content_root("") / ".gitignore"
                )
                if gitignore_path.exists():
                    with open(gitignore_path, "r") as gitignore_file:
                        gitignore_patterns = [
                            line.strip()
                            for line in gitignore_file
                            if line.strip() and not line.startswith("#")
                        ]
                        if "*" in gitignore_patterns:
                            gitignore_patterns.remove("*")

                files = []
                for file_type in self.file_types:
                    for root in self.content_roots:
                        content_root = Path(f"{temp_dir}/{self.get_content_root(root)}")
                        for file in content_root.rglob(file_type):
                            if should_include_file(
                                file.relative_to(content_root),
                                gitignore_patterns,
                                BLACKLIST,
                            ):
                                files.append(file)
                docs = []
                for file in files:
                    file_path = file.relative_to(temp_dir)
                    file_ext = file_path.suffix
                    source_type = SOURCE_TYPE_MAP.get(file_ext, "Document")
                    file_path = str(Path(*file_path.parts[1:]))
                    gh_file_path = f"{self.repo_url}/blob/{self.tag if self.tag else self.branch}/{file_path}"
                    docs.append(
                        {
                            "uri": gh_file_path,
                            "content": file.read_text(),
                            "source_type": source_type,
                        }
                    )
                return docs

    async def arun(self):
        docs = await self.extract_zip_data()
        output_docs = []
        for doc in docs:
            if doc["source_type"] != "Source Code":
                html_contents = doc["content"]
                if doc["source_type"] == "Notebook":
                    contents = read_notebook(doc["content"])
                    html_contents = md_to_html(contents)
                elif doc["source_type"] == "Webpage":
                    html_contents = md_to_html(doc["content"])
                html_obj = load_html_content(html_contents, doc["uri"])
                md_content = html_to_md(html_obj["content"])
                md_content = cleanup_text(md_content)
                output_doc = Document(
                    source_type=doc["source_type"],
                    uri=doc["uri"],
                    content=md_content,
                    num_tokens=length_fn(md_content),
                    links=html_obj["links"],
                )
            else:
                output_doc = Document(
                    source_type=doc["source_type"],
                    uri=doc["uri"],
                    content=doc["content"],
                    num_tokens=length_fn(doc["content"]),
                    links=[],
                )
            output_docs.append(output_doc)
        return output_docs


async def main():
    from tqdm import tqdm

    repos = [
        {
            "owner": "openai",
            "repo": "openai-cookbook",
            "content_roots": ["articles", "examples"],
        },
        {
            "owner": "anthropics",
            "repo": "anthropic-cookbook",
            "content_roots": [
                "misc",
                "multimodal",
                "skills",
                "third_party",
                "tool_use",
            ],
        },
        {"owner": "BerriAI", "repo": "litellm", "content_roots": ["cookbook"]},
        {
            "owner": "dair-ai",
            "repo": "Prompt-Engineering-Guide",
            "content_roots": ["notebooks"],
        },
        {"owner": "wandb", "repo": "weave", "content_roots": ["docs/notebooks"]},
        {"owner": "wandb", "repo": "edu", "content_roots": ["rag-advanced"]},
        {"owner": "cohere-ai", "repo": "notebooks", "content_roots": ["notebooks"]},
        {"owner": "jxnl", "repo": "instructor", "content_roots": ["examples"]},
        {
            "owner": "NirDiamant",
            "repo": "RAG_Techniques",
            "content_roots": ["all_rag_techniques", "evaluation"],
        },
    ]
    with open("data/cookbook_docs.jsonl", "w+") as f:
        for repo in tqdm(repos):
            async with RepoScrapper(**repo) as scrapper:
                docs = await scrapper.arun()
                for doc in docs:
                    line = doc.model_dump_json() + "\n"
                    f.write(line)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
