# LLM Pricing

Compare the pricing of various Large Language Models. Prices are shown in USD per 1M tokens.

> [!NOTE]
> Prices last updated on **2026-02-22**. All values are per 1 million tokens unless otherwise specified.
> Non-token-based models are marked with \* and converted as follows:
>
> - **Character-based** (TTS, Translation): $X per 1M chars → assume ~4 chars/token → multiply by 4. Cost split 50/50 input/output.
> - **Minute-based** (STT, Transcription): $X per minute → assume ~200 tokens/minute → multiply by 5000. Cost split 50/50 input/output.
>
> Search supports type keywords: `text`, `image`, `speech`, `transcription`, `translate`.

<style>
#pricingSearch {
    width: 100%; padding: 10px 14px; margin-bottom: 6px;
    border: 1px solid var(--md-default-fg-color--lightest, #ddd);
    border-radius: 6px; font-size: 15px; box-sizing: border-box;
    background: var(--md-default-bg-color, #fff);
    color: var(--md-default-fg-color, #333);
    transition: border-color 0.15s;
}
#pricingSearch:focus { outline: none; border-color: var(--md-primary-fg-color, #4285F4); }
.pricing-result-count {
    font-size: 12px; color: var(--md-default-fg-color--light, #888);
    margin-bottom: 16px; text-align: right;
}
.pricing-chart-wrap { width: 100%; height: 420px; margin-bottom: 6px; position: relative; }
.pricing-chart-btns { position: absolute; top: 8px; right: 8px; display: flex; gap: 6px; }
.pricing-chart-btn {
    padding: 3px 10px; font-size: 11px; cursor: pointer;
    border: 1px solid var(--md-default-fg-color--lightest, #ccc);
    border-radius: 4px;
    background: var(--md-default-bg-color, #fff);
    color: var(--md-default-fg-color--light, #666);
    transition: background 0.15s, color 0.15s;
}
.pricing-chart-btn:hover, .pricing-chart-btn.active {
    background: var(--md-primary-fg-color, #4285F4);
    color: #fff; border-color: transparent;
}
#pricingTable { width: 100%; border-collapse: collapse; font-size: 14px; }
#pricingTable thead tr {
    background-color: var(--md-primary-fg-color);
    color: var(--md-primary-bg-color);
}
#pricingTable th {
    padding: 10px 12px; text-align: left;
    border-bottom: 2px solid var(--md-default-fg-color--lightest, #ddd);
    cursor: pointer; user-select: none; white-space: nowrap;
    font-weight: 600; letter-spacing: 0.02em;
    transition: opacity 0.1s;
}
#pricingTable th:hover { opacity: 0.85; }
#pricingTable td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--md-default-fg-color--lightest, #eee);
    vertical-align: middle;
}
#pricingTable tbody tr:hover td { background: var(--md-code-bg-color, #f5f5f5); }
#pricingTable tbody tr:last-child td { border-bottom: none; }
.price-cell { font-variant-numeric: tabular-nums; font-family: 'Courier New', monospace; }
.vendor-chip {
    display: inline-block; padding: 2px 8px; border-radius: 99px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; white-space: nowrap;
    border: 1.5px solid currentColor; opacity: 0.85;
}
.empty-state {
    text-align: center; padding: 40px 20px;
    color: var(--md-default-fg-color--light, #999); font-style: italic;
}
</style>

<input type="text" id="pricingSearch" placeholder="🔍 Search models, vendors, or type (text / image / speech / transcription / translate)…">
<div class="pricing-result-count" id="resultCount"></div>

<div class="pricing-container">
    <div class="pricing-chart-wrap">
        <canvas id="pricingChart"></canvas>
        <div class="pricing-chart-btns">
            <button class="pricing-chart-btn" id="logToggle" onclick="toggleScale()">Log scale</button>
            <button class="pricing-chart-btn" id="resetZoom" onclick="if(chartInstance)chartInstance.resetZoom()">Reset zoom</button>
        </div>
    </div>
    <div id="chartLegend"></div>

    <table id="pricingTable">
        <thead>
            <tr>
                <th id="th-0" onclick="sortTable(0)">Name</th>
                <th id="th-1" onclick="sortTable(1)">Vendor</th>
                <th id="th-2" onclick="sortTable(2)">Input $/1M</th>
                <th id="th-3" onclick="sortTable(3)">Output $/1M</th>
            </tr>
        </thead>
        <tbody id="pricingBody">
            <!-- Data will be populated by JS -->
        </tbody>
    </table>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom"></script>
<script>
const pricingData = {
    "updated_at": "2026-02-22",
    "prices": [
        {"id": "amazon-nova-micro", "vendor": "amazon", "name": "Amazon Nova Micro", "input": 0.035, "output": 0.14},
        {"id": "amazon-nova-lite", "vendor": "amazon", "name": "Amazon Nova Lite", "input": 0.06, "output": 0.24},
        {"id": "amazon-nova-pro", "vendor": "amazon", "name": "Amazon Nova Pro", "input": 0.8, "output": 3.2},
        {"id": "amazon-nova-premier", "vendor": "amazon", "name": "Amazon Nova Premier", "input": 2.5, "output": 12.5},
        {"id": "amazon-nova-2-omni-preview", "vendor": "amazon", "name": "Amazon Nova 2 Omni (Preview)", "input": 0.3, "output": 2.5},
        {"id": "amazon-nova-2-pro-preview", "vendor": "amazon", "name": "Amazon Nova 2 Pro (Preview)", "input": 1.25, "output": 10.0},
        {"id": "amazon-nova-2-lite", "vendor": "amazon", "name": "Amazon Nova 2 Lite", "input": 0.3, "output": 2.5},
        {"id": "amazon.titan-image-generator-v2:0", "vendor": "amazon", "name": "Titan Image Generator v2*", "input": 5.0, "output": 5.0},
        {"id": "amazon.nova-canvas-v1:0", "vendor": "amazon", "name": "Amazon Nova Canvas v1*", "input": 20.0, "output": 20.0},
        {"id": "claude-3.7-sonnet", "vendor": "anthropic", "name": "Claude 3.7 Sonnet", "input": 3, "output": 15},
        {"id": "claude-3.5-sonnet", "vendor": "anthropic", "name": "Claude 3.5 Sonnet", "input": 3, "output": 15},
        {"id": "claude-3-opus", "vendor": "anthropic", "name": "Claude 3 Opus", "input": 15, "output": 75},
        {"id": "claude-3-haiku", "vendor": "anthropic", "name": "Claude 3 Haiku", "input": 0.25, "output": 1.25},
        {"id": "claude-3.5-haiku", "vendor": "anthropic", "name": "Claude 3.5 Haiku", "input": 0.8, "output": 4},
        {"id": "claude-4.5-haiku", "vendor": "anthropic", "name": "Claude 4.5 Haiku", "input": 1, "output": 5},
        {"id": "claude-sonnet-4.5", "vendor": "anthropic", "name": "Claude Sonnet 4 and 4.5 \u2264200k", "input": 3, "output": 15},
        {"id": "claude-sonnet-4.5-200k", "vendor": "anthropic", "name": "Claude Sonnet 4 and 4.5 >200k", "input": 6, "output": 22.5},
        {"id": "claude-sonnet-4.6", "vendor": "anthropic", "name": "Claude Sonnet 4.6 \u2264200k", "input": 3, "output": 15},
        {"id": "claude-sonnet-4.6-200k", "vendor": "anthropic", "name": "Claude Sonnet 4.6 >200k", "input": 6, "output": 22.5},
        {"id": "claude-opus-4", "vendor": "anthropic", "name": "Claude Opus 4", "input": 15, "output": 75},
        {"id": "claude-opus-4-1", "vendor": "anthropic", "name": "Claude Opus 4.1", "input": 15, "output": 75},
        {"id": "claude-opus-4-5", "vendor": "anthropic", "name": "Claude Opus 4.5", "input": 5, "output": 25},
        {"id": "claude-opus-4.6", "vendor": "anthropic", "name": "Claude Opus 4.6 \u2264200k", "input": 5, "output": 25},
        {"id": "claude-opus-4.6-200k", "vendor": "anthropic", "name": "Claude Opus 4.6 >200k", "input": 10, "output": 37.5},
        {"id": "gemini-2.5-pro-preview-03-25", "vendor": "google", "name": "Gemini 2.5 Pro Preview \u2264200k", "input": 1.25, "output": 10},
        {"id": "gemini-2.5-pro-preview-03-25-200k", "vendor": "google", "name": "Gemini 2.5 Pro Preview >200k", "input": 2.5, "output": 15},
        {"id": "gemini-2.0-flash-lite", "vendor": "google", "name": "Gemini 2.0 Flash Lite", "input": 0.08, "output": 0.3},
        {"id": "gemini-2.0-flash", "vendor": "google", "name": "Gemini 2.0 Flash", "input": 0.1, "output": 0.4},
        {"id": "gemini-2.5-flash", "vendor": "google", "name": "Gemini 2.5 Flash", "input": 0.3, "output": 2.5},
        {"id": "gemini-2.5-flash-lite", "vendor": "google", "name": "Gemini 2.5 Flash-Lite", "input": 0.1, "output": 0.4},
        {"id": "gemini-2.5-flash-preview-09-2025", "vendor": "google", "name": "Gemini 2.5 Flash Preview (09-2025)", "input": 0.3, "output": 2.5},
        {"id": "gemini-2.5-pro", "vendor": "google", "name": "Gemini 2.5 Pro \u2264200k", "input": 1.25, "output": 10},
        {"id": "gemini-2.5-pro-200k", "vendor": "google", "name": "Gemini 2.5 Pro >200k", "input": 2.5, "output": 15},
        {"id": "gemini-3.1-pro-preview", "vendor": "google", "name": "Gemini 3.1 Pro \u2264200k", "input": 2, "output": 12},
        {"id": "gemini-3.1-pro-preview-200k", "vendor": "google", "name": "Gemini 3.1 Pro >200k", "input": 4, "output": 18},
        {"id": "gemini-3-pro-preview", "vendor": "google", "name": "Gemini 3 Pro \u2264200k", "input": 2, "output": 12},
        {"id": "gemini-3-pro-preview-200k", "vendor": "google", "name": "Gemini 3 Pro >200k", "input": 4, "output": 18},
        {"id": "gemini-3-flash-preview", "vendor": "google", "name": "Gemini 3 Flash Preview", "input": 0.5, "output": 3},
        {"id": "gemini-2.5-flash-thinking", "vendor": "google", "name": "Gemini 2.5 Flash Thinking", "input": 0.3, "output": 2.5},
        {"id": "gemini-2.5-flash-lite-thinking", "vendor": "google", "name": "Gemini 2.5 Flash-Lite Thinking", "input": 0.1, "output": 0.4},
        {"id": "gemini-3-flash-preview-thinking", "vendor": "google", "name": "Gemini 3 Flash Preview Thinking", "input": 0.5, "output": 3.0},
        {"id": "gemini-2.5-flash-preview-tts", "vendor": "google", "name": "Gemini 2.5 Flash Preview TTS", "input": 0.5, "output": 10},
        {"id": "gemini-2.5-pro-preview-tts", "vendor": "google", "name": "Gemini 2.5 Pro Preview TTS", "input": 1, "output": 20},
        {"id": "gpt-4.5", "vendor": "openai", "name": "GPT-4.5", "input": 75, "output": 150},
        {"id": "gpt-4o", "vendor": "openai", "name": "GPT-4o", "input": 2.5, "output": 10},
        {"id": "gpt-4o-mini", "vendor": "openai", "name": "GPT-4o Mini", "input": 0.15, "output": 0.6},
        {"id": "chatgpt-4o-latest", "vendor": "openai", "name": "ChatGPT 4o Latest", "input": 5, "output": 15},
        {"id": "o1-preview", "vendor": "openai", "name": "o1 and o1-preview", "input": 15, "output": 60},
        {"id": "o1-pro", "vendor": "openai", "name": "o1 Pro", "input": 150, "output": 600},
        {"id": "o1-mini", "vendor": "openai", "name": "o1-mini", "input": 1.1, "output": 4.4},
        {"id": "o3-mini", "vendor": "openai", "name": "o3-mini", "input": 1.1, "output": 4.4},
        {"id": "gpt-4.1", "vendor": "openai", "name": "GPT-4.1", "input": 2, "output": 8},
        {"id": "gpt-4.1-mini", "vendor": "openai", "name": "GPT-4.1 Mini", "input": 0.4, "output": 1.6},
        {"id": "gpt-4.1-nano", "vendor": "openai", "name": "GPT-4.1 Nano", "input": 0.1, "output": 0.4},
        {"id": "o3", "vendor": "openai", "name": "o3", "input": 2, "output": 8},
        {"id": "gpt-5.3-codex", "vendor": "openai", "name": "GPT-5.3 Codex", "input": 1.25, "output": 10.0},
        {"id": "codex-mini-latest", "vendor": "openai", "name": "Codex Mini Latest", "input": 1.5, "output": 6.0},
        {"id": "gpt-5.3", "vendor": "openai", "name": "GPT-5.3 (Preview)", "input": 1.5, "output": 7.5},
        {"id": "o4-mini", "vendor": "openai", "name": "o4-mini", "input": 1.1, "output": 4.4},
        {"id": "gpt-5-nano", "vendor": "openai", "name": "GPT-5 Nano", "input": 0.05, "output": 0.4},
        {"id": "gpt-5-mini", "vendor": "openai", "name": "GPT-5 Mini", "input": 0.25, "output": 2},
        {"id": "gpt-5", "vendor": "openai", "name": "GPT-5", "input": 1.25, "output": 10},
        {"id": "gpt-image-1", "vendor": "openai", "name": "gpt-image-1 (image gen)", "input": 10, "output": 40},
        {"id": "gpt-image-1-mini", "vendor": "openai", "name": "gpt-image-1-mini (image gen)", "input": 2, "output": 8},
        {"id": "gpt-image-1.5", "vendor": "openai", "name": "gpt-image-1.5 (image gen)", "input": 5, "output": 34},
        {"id": "gpt-5-pro", "vendor": "openai", "name": "GPT-5 Pro", "input": 15, "output": 120},
        {"id": "o3-pro", "vendor": "openai", "name": "o3 Pro", "input": 20, "output": 80},
        {"id": "o4-mini-deep-research", "vendor": "openai", "name": "o4-mini Deep Research", "input": 2, "output": 8},
        {"id": "o3-deep-research", "vendor": "openai", "name": "o3 Deep Research", "input": 10, "output": 40},
        {"id": "gpt-5.1-codex-mini", "vendor": "openai", "name": "GPT-5.1 Codex mini", "input": 0.25, "output": 2.0},
        {"id": "gpt-5.1-codex", "vendor": "openai", "name": "GPT-5.1 Codex", "input": 1.25, "output": 10.0},
        {"id": "gpt-5.1", "vendor": "openai", "name": "GPT-5.1", "input": 1.25, "output": 10.0},
        {"id": "gpt-5.2", "vendor": "openai", "name": "GPT-5.2", "input": 1.75, "output": 14.0},
        {"id": "gpt-5.2-pro", "vendor": "openai", "name": "GPT-5.2 Pro", "input": 21.0, "output": 168.0},
        {"id": "gemini-3-pro-image-preview", "vendor": "google", "name": "Gemini 3 Pro Image Preview", "input": 2.0, "output": 120.0},
        {"id": "gemini-2.5-flash-image", "vendor": "google", "name": "Gemini 2.5 Flash Image*", "input": 15.0, "output": 15.0},
        {"id": "imagen-4", "vendor": "google", "name": "Imagen 4*", "input": 20.0, "output": 20.0},
        {"id": "imagen-4-fast", "vendor": "google", "name": "Imagen 4 Fast*", "input": 10.0, "output": 10.0},
        {"id": "imagen-4-ultra", "vendor": "google", "name": "Imagen 4 Ultra*", "input": 30.0, "output": 30.0},
        {"id": "imagen-4.0-ultra-generate-001", "vendor": "google", "name": "Imagen 4.0 Ultra Generate 001*", "input": 30.0, "output": 30.0},
        {"id": "imagen-4.0-fast-generate-001", "vendor": "google", "name": "Imagen 4.0 Fast Generate 001*", "input": 10.0, "output": 10.0},
        {"id": "chirp_3", "vendor": "google", "name": "GCP Chirp 3 (STT)*", "input": 40.0, "output": 40.0},
        {"id": "aws-polly", "vendor": "amazon", "name": "AWS Polly*", "input": 32.0, "output": 32.0},
        {"id": "aws-transcribe", "vendor": "amazon", "name": "AWS Transcribe*", "input": 60.0, "output": 60.0},
        {"id": "gcp-chirp3-tts", "vendor": "google", "name": "GCP Chirp 3 (TTS)*", "input": 32.0, "output": 32.0},
        {"id": "aws-translate", "vendor": "amazon", "name": "AWS Translate*", "input": 30.0, "output": 30.0},
        {"id": "gcp-translate", "vendor": "google", "name": "GCP Cloud Translation*", "input": 40.0, "output": 40.0}
    ]
};

// Track the currently displayed (filtered + sorted) data
let currentData = [...pricingData.prices];
// Active type filter set by clicking legend type items (null = show all)
let activeTypeFilter = null;
// Log scale toggle
let useLogScale = false;
// Sort state
let sortCol = -1, sortDir = 1;

const VENDOR_COLORS = {
    'google':    '#8B5CF6',
    'amazon':    '#FFB800',
    'openai':    '#1A1A1A',
    'anthropic': '#E8511A'
};

function formatPrice(val) {
    if (val === 0) return '$0';
    if (val < 0.1)  return '$' + val.toFixed(4).replace(/\.?0+$/, '');
    if (val < 1)    return '$' + val.toFixed(3).replace(/\.?0+$/, '');
    if (val < 10)   return '$' + val.toFixed(2).replace(/\.?0+$/, '');
    if (val < 100)  return '$' + val.toFixed(1).replace(/\.?0+$/, '');
    return '$' + Math.round(val);
}

function populateTable(data) {
    const body = document.getElementById('pricingBody');
    body.innerHTML = '';
    if (data.length === 0) {
        body.innerHTML = '<tr><td colspan="4" class="empty-state">No models match your search.</td></tr>';
        return;
    }
    data.forEach(item => {
        const color = VENDOR_COLORS[item.vendor] || '#888';
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.name}</td>
            <td><span class="vendor-chip" style="color:${color};border-color:${color}">${item.vendor}</span></td>
            <td class="price-cell">${formatPrice(item.input)}</td>
            <td class="price-cell">${formatPrice(item.output)}</td>
        `;
        body.appendChild(row);
    });
}

function updateResultCount(data) {
    const el = document.getElementById('resultCount');
    const total = pricingData.prices.length;
    el.textContent = data.length === total
        ? `${total} models`
        : `${data.length} of ${total} models`;
}

function getModelType(item) {
    const n = item.name.toLowerCase();
    const id = item.id.toLowerCase();
    if (n.includes('image') || n.includes('canvas') || n.includes('imagen')) return 'image';
    if (n.includes('tts') || n.includes('polly')) return 'speech';
    if (n.includes('stt') || n.includes('transcribe')) return 'transcription';
    if (n.includes('translate') || id.includes('translate')) return 'translate';
    return 'text';
}

function getFilteredData() {
    const searchTerm = document.getElementById('pricingSearch').value.toLowerCase();
    return pricingData.prices.filter(item => {
        const matchesSearch = !searchTerm ||
            item.name.toLowerCase().includes(searchTerm) ||
            item.vendor.toLowerCase().includes(searchTerm) ||
            item.id.toLowerCase().includes(searchTerm) ||
            getModelType(item).includes(searchTerm);
        const matchesType = !activeTypeFilter || getModelType(item) === activeTypeFilter;
        return matchesSearch && matchesType;
    });
}

function filterTable() {
    currentData = getFilteredData();
    populateTable(currentData);
    updateResultCount(currentData);
    renderChart(currentData);
}

function toggleTypeFilter(type) {
    activeTypeFilter = activeTypeFilter === type ? null : type;
    filterTable();
    buildLegend();
}

function sortTable(columnIndex) {
    const propertyMap = ['name', 'vendor', 'input', 'output'];
    const property = propertyMap[columnIndex];

    if (sortCol === columnIndex) {
        sortDir *= -1;
    } else {
        sortCol = columnIndex;
        sortDir = 1;
    }

    currentData.sort((a, b) => {
        const valA = a[property], valB = b[property];
        return (typeof valA === 'string' ? valA.localeCompare(valB) : valA - valB) * sortDir;
    });

    // Update header arrows
    for (let i = 0; i < 4; i++) {
        const th = document.getElementById('th-' + i);
        const labels = ['Name', 'Vendor', 'Input $/1M', 'Output $/1M'];
        th.textContent = labels[i] + (i === sortCol ? (sortDir === 1 ? ' ↑' : ' ↓') : '');
    }

    populateTable(currentData);
    renderChart(currentData);
}

function toggleScale() {
    useLogScale = !useLogScale;
    const btn = document.getElementById('logToggle');
    btn.classList.toggle('active', useLogScale);
    btn.textContent = useLogScale ? 'Linear scale' : 'Log scale';
    renderChart(currentData);
}

// Chart.js Visualization
let chartInstance = null;

function renderChart(data) {
    const ctx = document.getElementById('pricingChart').getContext('2d');

    const datasets = data.map(item => {
        const color = VENDOR_COLORS[item.vendor] || '#888';
        const modelType = getModelType(item);
        let pointStyle = 'circle';
        if (modelType === 'image') pointStyle = 'star';
        else if (modelType === 'speech') pointStyle = 'rectRot';
        else if (modelType === 'transcription') pointStyle = 'triangle';
        else if (modelType === 'translate') pointStyle = 'crossRot';

        return {
            label: item.name,
            data: [{x: item.input, y: item.output}],
            backgroundColor: color + 'cc',
            borderColor: color,
            borderWidth: 1.5,
            pointStyle: pointStyle,
            pointRadius: 6,
            pointHoverRadius: 9
        };
    });

    if (chartInstance) chartInstance.destroy();

    const scaleType = useLogScale ? 'logarithmic' : 'linear';

    chartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(20,20,20,0.9)',
                    titleColor: '#fff',
                    bodyColor: '#ccc',
                    padding: 10,
                    cornerRadius: 6,
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}`,
                        afterLabel: ctx => `  Input: ${formatPrice(ctx.raw.x)}  ·  Output: ${formatPrice(ctx.raw.y)}`
                    }
                },
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Input vs Output cost ($ per 1M tokens)',
                    color: 'var(--md-default-fg-color--light, #666)',
                    font: { size: 13, weight: 'normal' },
                    padding: { bottom: 12 }
                },
                zoom: {
                    pan: { enabled: true, mode: 'xy' },
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }
                }
            },
            scales: {
                x: {
                    type: scaleType,
                    position: 'bottom',
                    title: { display: true, text: 'Input ($/1M tokens)', color: 'var(--md-default-fg-color--light, #666)' },
                    grid: { color: 'rgba(128,128,128,0.12)' },
                    ...(useLogScale ? {} : { beginAtZero: true })
                },
                y: {
                    type: scaleType,
                    title: { display: true, text: 'Output ($/1M tokens)', color: 'var(--md-default-fg-color--light, #666)' },
                    grid: { color: 'rgba(128,128,128,0.12)' },
                    ...(useLogScale ? {} : { beginAtZero: true })
                }
            }
        }
    });
}

function buildLegend() {
    const el = document.getElementById('chartLegend');
    const vendorItems = [
        {color: '#8B5CF6', label: 'Google'},
        {color: '#FFB800', label: 'Amazon'},
        {color: '#1A1A1A', label: 'OpenAI'},
        {color: '#E8511A', label: 'Anthropic'}
    ];
    const typeItems = [
        {symbol: '●', label: 'Text',          type: 'text'},
        {symbol: '★', label: 'Image',         type: 'image'},
        {symbol: '▲', label: 'Transcription', type: 'transcription'},
        {symbol: '◆', label: 'Speech',        type: 'speech'},
        {symbol: '✕', label: 'Translate',     type: 'translate'}
    ];
    let html = '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:20px;margin-bottom:8px;font-size:14px;">';
    html += '<strong style="color:#666">Vendor:</strong> ';
    vendorItems.forEach(v => { html += `<span style="color:${v.color}">${v.label}</span>`; });
    html += '</div>';
    html += '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:20px;margin-bottom:24px;font-size:14px;">';
    html += '<strong style="color:#666">Type:</strong> ';
    typeItems.forEach(t => {
        const isActive = activeTypeFilter === t.type;
        const style = `cursor:pointer;padding:2px 8px;border-radius:4px;user-select:none;${isActive ? 'background:#e0e0e0;font-weight:bold;' : ''}`;
        html += `<span style="${style}" onclick="toggleTypeFilter('${t.type}')" title="Click to filter">${t.symbol} ${t.label}</span>`;
    });
    if (activeTypeFilter) {
        html += `<span style="color:#666;margin-left:8px;font-style:italic;cursor:pointer;" onclick="toggleTypeFilter(activeTypeFilter)">✕ clear filter</span>`;
    }
    html += '<span style="color:#999;margin-left:12px;font-style:italic">(scroll/pinch to zoom, drag to pan)</span>';
    html += '</div>';
    el.innerHTML = html;
}

// Double-click to reset zoom
document.getElementById('pricingChart').addEventListener('dblclick', function() {
    if (chartInstance) chartInstance.resetZoom();
});

document.getElementById('pricingSearch').addEventListener('input', filterTable);
buildLegend();
populateTable(pricingData.prices);
updateResultCount(pricingData.prices);
renderChart(pricingData.prices);
</script>
