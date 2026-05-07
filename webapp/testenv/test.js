const puppeteer = require('puppeteer');
(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    await page.goto('http://localhost:8001');
    await new Promise(r => setTimeout(r, 1000));
    try {
        await page.click('button[onclick="startMonitoringSession()"]');
        console.log('Clicked home button');
    } catch(e) { console.log('Click error:', e.message); }
    
    await new Promise(r => setTimeout(r, 2000));
    await page.screenshot({path: 'screen.png'});
    
    // Evaluate if btn-start was clicked
    const text = await page.$eval('#btn-start', el => el.innerText);
    console.log('btn-start text:', text);
    
    await browser.close();
})();
