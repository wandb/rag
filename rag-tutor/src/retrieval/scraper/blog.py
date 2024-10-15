import hashlib
import json
import os
import time

import httpx
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_crawler_strategy import (
    AsyncCrawlResponse,
    AsyncPlaywrightCrawlerStrategy,
)
from models import Document
from playwright.async_api import Browser, Error, Page
from utils import cleanup_text, html_to_md, length_fn, load_html_content

readability_script_url = (
    "https://cdnjs.cloudflare.com/ajax/libs/readability/0.5.0/Readability.js"
)
response = httpx.get(readability_script_url)
readability_js_content = response.text

# Mozilla's Readability script to inject
readability_js = """
{readability_js_content}
(function() {{
    var documentClone = document.cloneNode(true);
    var reader = new Readability(documentClone);
    var article = reader.parse();
    if (article) {{
        return `<html><body><h1>${{article.title}}</h1>${{article.content}}</body></html>`;
    }} else {{
        return document.documentElement.outerHTML;
    }}
}})();
""".format(
    readability_js_content=readability_js_content
)


class CSPCompliantAsyncPlaywrightCrawlerStrategy(AsyncPlaywrightCrawlerStrategy):
    async def crawl(self, url: str, **kwargs) -> AsyncCrawlResponse:
        response_headers = {}
        status_code = None

        self._cleanup_expired_sessions()
        session_id = kwargs.get("session_id")
        if session_id:
            context, page, _ = self.sessions.get(session_id, (None, None, None))
            if not context:
                context = await self.browser.new_context(
                    bypass_csp=True,
                    user_agent=self.user_agent,
                    proxy={"server": self.proxy} if self.proxy else None,
                )
                await context.set_extra_http_headers(self.headers)
                page = await context.new_page()
                self.sessions[session_id] = (context, page, time.time())
        else:
            context = await self.browser.new_context(
                user_agent=self.user_agent,
                bypass_csp=True,
                proxy={"server": self.proxy} if self.proxy else None,
            )
            await context.set_extra_http_headers(self.headers)
            page = await context.new_page()

        try:
            if self.verbose:
                print(
                    f"[LOG] 🕸️ Crawling {url} using AsyncPlaywrightCrawlerStrategy..."
                )

            if self.use_cached_html:
                cache_file_path = os.path.join(
                    Path.home(),
                    ".crawl4ai",
                    "cache",
                    hashlib.md5(url.encode()).hexdigest(),
                )
                if os.path.exists(cache_file_path):
                    html = ""
                    with open(cache_file_path, "r") as f:
                        html = f.read()
                    # retrieve response headers and status code from cache
                    with open(cache_file_path + ".meta", "r") as f:
                        meta = json.load(f)
                        response_headers = meta.get("response_headers", {})
                        status_code = meta.get("status_code")
                    response = AsyncCrawlResponse(
                        html=html,
                        response_headers=response_headers,
                        status_code=status_code,
                    )
                    return response

            if not kwargs.get("js_only", False):
                await self.execute_hook("before_goto", page)
                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=60000
                )
                await self.execute_hook("after_goto", page)

                # Get status code and headers
                status_code = response.status
                response_headers = response.headers
            else:
                status_code = 200
                response_headers = {}

            await page.wait_for_selector("body")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            js_code = kwargs.get("js_code", kwargs.get("js", self.js_code))
            if js_code:
                if isinstance(js_code, str):
                    r = await page.evaluate(js_code)
                elif isinstance(js_code, list):
                    for js in js_code:
                        await page.evaluate(js)

                # await page.wait_for_timeout(100)
                await page.wait_for_load_state("networkidle")
                # Check for on execution even
                await self.execute_hook("on_execution_started", page)

            wait_for = kwargs.get("wait_for")
            if wait_for:
                try:
                    await self.smart_wait(
                        page, wait_for, timeout=kwargs.get("timeout", 30000)
                    )
                except Exception as e:
                    raise RuntimeError(f"Wait condition failed: {str(e)}")

            html = await page.content()
            page = await self.execute_hook("before_return_html", page, html)

            if self.verbose:
                print(f"[LOG] ✅ Crawled {url} successfully!")

            if self.use_cached_html:
                cache_file_path = os.path.join(
                    Path.home(),
                    ".crawl4ai",
                    "cache",
                    hashlib.md5(url.encode()).hexdigest(),
                )
                with open(cache_file_path, "w", encoding="utf-8") as f:
                    f.write(html)
                # store response headers and status code in cache
                with open(cache_file_path + ".meta", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "response_headers": response_headers,
                            "status_code": status_code,
                        },
                        f,
                    )

            response = AsyncCrawlResponse(
                html=html, response_headers=response_headers, status_code=status_code
            )
            return response
        except Error as e:
            raise Error(f"Failed to crawl {url}: {str(e)}")
        finally:
            if not session_id:
                await page.close()

        # try:
        #     html = await _crawl()
        #     return sanitize_input_encode(html)
        # except Error as e:
        #     raise Error(f"Failed to crawl {url}: {str(e)}")
        # except Exception as e:
        #     raise Exception(f"Failed to crawl {url}: {str(e)}")


async def on_browser_created(browser: Browser):
    print("[HOOK] on_browser_created")
    # Example customization: set browser viewport size
    context = await browser.new_context(bypass_csp=True)
    page = await context.new_page()
    await page.close()
    await context.close()


async def on_execution_started(page: Page):
    print("[HOOK] on_execution_started")
    # Example customization: perform actions after JS execution
    result = await page.evaluate(readability_js)
    await page.set_content(html=result)


async def load_blog_article(url: str) -> Document:
    crawler_strategy = CSPCompliantAsyncPlaywrightCrawlerStrategy(verbose=True)
    crawler_strategy.set_hook("on_browser_created", on_browser_created)
    crawler_strategy.set_hook("on_execution_started", on_execution_started)

    async with AsyncWebCrawler(
        verbose=True, crawler_strategy=crawler_strategy
    ) as crawler:
        result = await crawler.arun(
            url,
            js_code=readability_js,
        )
        html_obj = load_html_content(result.html, url)
        md_content = html_to_md(html_obj["content"])
        md_content = cleanup_text(md_content)
        return Document(
            content=md_content,
            uri=html_obj["uri"],
            links=html_obj["links"],
            num_tokens=length_fn(md_content),
            source_type="Webpage",
        )


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from tqdm import tqdm

    with open("data/blog_articles.jsonl", "w+") as f:
        articles = Path("blog_sources.txt").read_text().splitlines()
        for article in tqdm(articles, total=len(articles)):
            article = asyncio.run(load_blog_article(article.strip()))
            line = article.model_dump_json() + "\n"
            f.write(line)
