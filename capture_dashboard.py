import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        # Ensure we wait for network idle to let Three.js or WebGL load
        await page.goto('http://localhost:8080', wait_until='networkidle')
        await page.wait_for_timeout(2000) # Give extra time for rendering
        await page.screenshot(path='ui_dashboard_mockup.png', full_page=True)
        await browser.close()
        print("Dashboard mockup saved to ui_dashboard_mockup.png")

asyncio.run(main())
