// Browser tests for the frontend runtime (#181). `npm run test:e2e`.
// The app serves the pages; tests/e2e/stubs.js answers every data request, so
// no SPARQL endpoint is needed (see tests/e2e/serve.py).
const { defineConfig } = require('@playwright/test');

const PORT = process.env.E2E_PORT || '5055';

module.exports = defineConfig({
    testDir: 'tests/e2e',
    timeout: 60000,
    retries: process.env.CI ? 1 : 0,
    reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
    use: {
        baseURL: `http://127.0.0.1:${PORT}`,
        viewport: { width: 1400, height: 900 },
    },
    projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
    webServer: {
        command: 'python tests/e2e/serve.py',
        url: `http://127.0.0.1:${PORT}/`,
        reuseExistingServer: !process.env.CI,
        timeout: 120000,
        env: { E2E_PORT: PORT },
    },
});
