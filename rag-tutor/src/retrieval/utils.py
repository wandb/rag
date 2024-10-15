import hashlib
import re
import uuid
from functools import partial
from typing import List, Tuple, Union
from urllib.parse import urljoin
from xml import etree
from xml.etree import ElementTree

import ftfy
import litellm
import markdown
import shortuuid
import tiktoken
from bs4 import BeautifulSoup, Tag
from litellm import aembedding
from litellm.caching import Cache
from lxml import etree
from markdownify import markdownify

litellm.cache = Cache(type="disk", disk_cache_dir="data/cache/litellm")

encoding = tiktoken.encoding_for_model("gpt-4o")
special_tokens_set = encoding.special_tokens_set


html_to_md = partial(
    markdownify,
    heading_style="ATX",
    escape_asterisks=False,
    escape_underscores=False,
    escape_misc=False,
)
md_to_html = partial(
    markdown.markdown,
    extensions=[
        "toc",
        "tables",
        "pymdownx.extra",
        "pymdownx.blocks.admonition",
        "pymdownx.magiclink",
        "pymdownx.blocks.tab",
        "pymdownx.pathconverter",
        "pymdownx.saneheaders",
        "pymdownx.striphtml",
        "pymdownx.highlight",
        "pymdownx.pathconverter",
        "pymdownx.escapeall",
        "pymdownx.snippets",
        "pymdownx.tasklist",
        "pymdownx.tilde",
        "pymdownx.mark",
        "pymdownx.inlinehilite",
        "pymdownx.emoji",
        "pymdownx.caret",
        "pymdownx.critic",
        "pymdownx.details",
        "pymdownx.superfences",
        "pymdownx.tabbed",
    ],
)


def generate_key(contents: str):
    hash_object = hashlib.sha256(contents.encode())
    hash_hex = hash_object.hexdigest()
    uuid_from_hash = uuid.UUID(hash_hex[:32])

    return shortuuid.encode(uuid_from_hash)


def get_main_content(soup: BeautifulSoup):
    # Check for <article> tag
    article = soup.find("article")
    if article:
        md_content = article.find("div", class_="theme-doc-markdown markdown")
        if md_content:
            return md_content
        return article

    # Check for <div class="notebody">
    notes_body = soup.find("div", class_="notebody")
    if notes_body:
        return notes_body

    # Check for <div class="notes">
    notes_div = soup.find("div", class_="notes")
    if notes_div:
        return notes_div

    # Check for <"main">
    main_div = soup.find("main")
    if main_div:
        return main_div

    # If none of the above are found, return the whole body
    return soup.find("body") or soup


def extract_links(soup: BeautifulSoup, url: str):
    links = {"internal": [], "external": []}
    url_base = url.split("/")[2]

    for a in soup.find_all("a", href=True):
        href = a["href"]
        link_data = {"href": href, "text": a.get_text(strip=True)}
        if href.startswith("http") and url_base not in href:
            links["external"].append(link_data)
        else:
            link_data["href"] = urljoin(url, link_data["href"])
            links["internal"].append(link_data)

    return links


def handle_admonitions(soup: BeautifulSoup):
    admonition_types = {
        "info": "NOTE",
        "tip": "TIP",
        "important": "IMPORTANT",
        "warning": "WARNING",
        "caution": "CAUTION",
    }

    for admonition in soup.find_all("div", class_="theme-admonition"):
        admonition_type = next(
            (t for t in admonition["class"] if t.startswith("theme-admonition-")), ""
        )
        admonition_type = admonition_type.replace("theme-admonition-", "")
        gfm_type = admonition_types.get(admonition_type, admonition_type.upper())

        content = admonition.find(
            "div", class_=lambda x: x and "admonitionContent" in x
        )

        new_html = f"> [!{gfm_type}]\n"
        if content:
            for element in content.children:
                if isinstance(element, Tag):
                    new_html += f"> {element.decode_contents().strip()}\n"
                elif element.strip():
                    new_html += f"> {element.strip()}\n"

        admonition.replace_with(BeautifulSoup(new_html, "html.parser"))

    return soup


def handle_tabs(soup: BeautifulSoup):
    for tab_container in soup.find_all("div", class_="tabs-container"):
        content_container = tab_container.find("div", class_="margin-top--md")
        tab_contents = (
            content_container.find_all("div", recursive=False)
            if content_container
            else []
        )
        new_html = "<ul>"
        for content in tab_contents:
            new_html += f"<center>\n{content.decode_contents()}\n</center><hr>"
        tab_container.replace_with(BeautifulSoup(new_html, "html.parser"))
    return soup


def replace_a_tags(soup: BeautifulSoup):
    for a in soup.find_all("a"):
        if a.string:
            a.replace_with(f"{a.string} ")
    return soup


def remove_diagrams_and_images(soup: BeautifulSoup):
    for svg in soup.find_all("svg"):
        svg.decompose()
    for img in soup.find_all("img"):
        if img.get("src", "").startswith("data:image"):
            img.decompose()
    for img in soup.find_all("img"):
        img.decompose()
    return soup


def remove_unwanted_tags(html_text: str):
    html_parser = etree.HTMLParser(remove_comments=True)
    root = etree.fromstring(html_text.encode("utf-8"), html_parser)
    etree.strip_elements(
        root,
        ["del", "img", "link", "noscript", "script", "style", "header", "footer"],
        with_tail=False,
    )
    if (main := root.find(".//main")) is not None:
        return ElementTree.tostring(main, encoding="utf-8")
    if (body := root.find(".//body")) is not None:
        return ElementTree.tostring(body, encoding="utf-8")
    return ElementTree.tostring(root, encoding="utf-8")


def clean_html_content(soup: BeautifulSoup):
    soup = remove_diagrams_and_images(soup)
    soup = handle_tabs(soup)
    soup = handle_admonitions(soup)
    soup = replace_a_tags(soup)
    return soup


def load_html_content(content: str, url: str) -> dict:
    soup = BeautifulSoup(content, "html.parser")
    article = get_main_content(soup)
    links = extract_links(article, url)
    article = clean_html_content(article)
    article = f"<html><body>{article}</body></html>"
    return {
        "content": article,
        "links": links,
        "uri": url,
    }


def length_fn(text: str) -> int:
    return len(encoding.encode(text, allowed_special="all"))


def remove_special_tokens(text: str) -> str:
    for token in special_tokens_set:
        text = text.replace(token, "")
    return text


ListLike = Union[List, Tuple]


def flatten_sequence(sequence: ListLike) -> ListLike:
    seq_type = type(sequence)
    return seq_type([item for sublist in sequence for item in sublist])


def cleanup_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = remove_special_tokens(text)
    text = text.replace("Äã", "")
    return text.strip()


async def embed_documents(documents, model="text-embedding-3-small", dimensions=512):
    response = await aembedding(model=model, input=documents, dimensions=dimensions)
    vectors = [item["embedding"] for item in response.data]
    return vectors
