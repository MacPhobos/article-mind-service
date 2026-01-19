"""JavaScript-rendered content extraction using Playwright."""

import time
from typing import Any

from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from .base import BaseExtractor, ExtractionResult
from .html_extractor import HTMLExtractor


class JSExtractor(BaseExtractor):
    """Extract content from JavaScript-rendered pages using Playwright.

    Design Decision: Use Playwright over Selenium
    - 35-45% faster execution than Selenium
    - Official Python support with async/await
    - Better auto-wait mechanisms

    Trade-offs:
    - ✅ Fast JS rendering (2-5s typical)
    - ✅ Handles SPAs and lazy-loading
    - ✅ Network interception for optimization
    - ❌ Higher resource usage than static extraction
    - ❌ Requires browser installation
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        block_resources: bool = True,
    ) -> None:
        """Initialize JS extractor.

        Args:
            headless: Run browser in headless mode
            timeout_ms: Navigation timeout in milliseconds
            block_resources: Block images/fonts to speed up loading
        """
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.block_resources = block_resources
        self._html_extractor = HTMLExtractor()

    def can_extract(self, content_type: str) -> bool:
        """JS extractor handles HTML that needs rendering."""
        return content_type.lower() in ("html", "text/html", "js-html")

    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract content by rendering the page with Playwright.

        Note: For JS extraction, we ignore the content parameter and
        fetch fresh from the URL with JavaScript execution.

        Args:
            content: Ignored (we fetch fresh)
            url: URL to render and extract

        Returns:
            ExtractionResult with rendered content
        """
        start_time = time.perf_counter()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)

            try:
                page = await browser.new_page()

                # Block unnecessary resources for speed
                if self.block_resources:

                    async def abort_route(route: Any) -> None:
                        await route.abort()

                    await page.route(
                        "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,eot,ico}", abort_route
                    )

                # Navigate and wait for content
                try:
                    await page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=self.timeout_ms,
                    )
                except PlaywrightTimeout:
                    # Try with domcontentloaded if networkidle times out
                    await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self.timeout_ms,
                    )

                # Wait a bit more for lazy-loaded content
                await page.wait_for_timeout(1000)

                # Get rendered HTML
                html = await page.content()

                # Extract using HTML extractor
                result = await self._html_extractor.extract(html, url)
                result.extraction_method = f"playwright+{result.extraction_method}"
                result.metadata["rendered"] = True

            finally:
                await browser.close()

        result.extraction_time_ms = (time.perf_counter() - start_time) * 1000
        return result
