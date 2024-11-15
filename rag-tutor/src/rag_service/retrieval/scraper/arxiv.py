import pypandoc
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from models import Document
from utils import cleanup_text, length_fn, load_html_content


def get_article_content(soup: BeautifulSoup) -> BeautifulSoup:
    article = soup.find("article", class_="ltx_document ltx_authors_1line")
    if article:
        return article
    else:
        return soup


def get_title(article: BeautifulSoup) -> str:
    title = article.find("h1", class_="ltx_title")
    if title:
        return title.prettify()
    else:
        return ""


def get_abstract(article: BeautifulSoup) -> str:
    abstract = article.find("div", class_="ltx_abstract")
    if abstract:
        abstract = abstract.find("p", class_="ltx_p")
        return f"<h2>Abstract</h2>\n\n{abstract.prettify()}"
    else:
        return ""


def get_sections(article: BeautifulSoup) -> str:
    sections = []
    for section in article.find_all("section", class_="ltx_section"):
        sections.append(section.prettify())
    return "".join(sections)


def parse_arxiv_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    for button in soup.find_all("button"):
        button.decompose()

    article = get_article_content(soup)
    title = get_title(article)
    abstract = get_abstract(article)
    sections = get_sections(article)

    article = f"""<article>
        <title>
            {title}
        </title>
        <abstract>
            {abstract}
        </abstract>
        <sections>
            {sections}
        </sections>
    </article>"""
    return article


def convert_to_md(html_content):
    # Specify the correct input and output formats with extensions
    input_format = (
        "html+tex_math_dollars+tex_math_single_backslash+tex_math_double_backslash"
    )
    output_format = "gfm+tex_math_dollars-raw_html"

    # Remove redundant -f and -t from extra_args
    pdoc_args = ["--mathjax", "--wrap=none"]

    # Perform the conversion
    converted = pypandoc.convert_text(
        html_content, output_format, format=input_format, extra_args=pdoc_args
    )
    return converted


async def load_arxiv_article(url: str) -> Document:
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url)
        article = parse_arxiv_html(result.html)
        html_obj = load_html_content(article, url)
        md_content = convert_to_md(html_obj["content"])
        md_content = cleanup_text(md_content)
        return Document(
            content=md_content,
            uri=html_obj["uri"],
            links=html_obj["links"],
            num_tokens=length_fn(md_content),
            source_type="Paper",
        )


if __name__ == "__main__":
    import asyncio
    import time
    from pathlib import Path

    from tqdm import tqdm

    with open("data/arxiv_articles.jsonl", "a+") as f:
        articles = Path("arxiv_sources.txt").read_text().splitlines()[35:]
        for article in tqdm(articles, total=len(articles)):
            article = asyncio.run(load_arxiv_article(article.strip()))
            line = article.model_dump_json() + "\n"
            f.write(line)
            time.sleep(30)
