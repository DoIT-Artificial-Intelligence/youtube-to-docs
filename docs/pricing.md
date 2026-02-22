# LLM Pricing

Compare the pricing of various Large Language Models. Prices are shown in USD per 1M tokens.

> [!NOTE]
> Prices last updated on **2026-02-06**. All values are per 1 million tokens unless otherwise specified.
> required for non-token based models (marked with \*):
>
> - **Audio/Character Pricing**: Models priced by minute or character are converted to "per 1M tokens" assuming ~4 chars/token or ~200 tokens/minute. The estimated cost is split 50/50 between input and output for comparison.

<input type="text" id="pricingSearch" placeholder="🔍 Search models or vendors..." style="width: 100%; padding: 12px 16px; margin-bottom: 24px; border: 1px solid #ccc; border-radius: 8px; font-size: 16px; box-sizing: border-box;">

<div class="pricing-container">
    <div style="width: 100%; height: 450px; margin-bottom: 8px; position: relative;">
        <canvas id="pricingChart"></canvas>
        <button id="resetZoom" onclick="if(chartInstance)chartInstance.resetZoom()" style="position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;cursor:pointer;border:1px solid #ccc;border-radius:4px;background:#fff;color:#666;">Reset Zoom</button>
    </div>
    <div id="chartLegend"></div>

    <table id="pricingTable" style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: var(--md-primary-fg-color); color: var(--md-primary-bg-color);">
                <th onclick="sortTable(0)" style="cursor: pointer; padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">Name ⇅</th>
                <th onclick="sortTable(1)" style="cursor: pointer; padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">Vendor ⇅</th>
                <th onclick="sortTable(2)" style="cursor: pointer; padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">Input ($/1M) ⇅</th>
                <th onclick="sortTable(3)" style="cursor: pointer; padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">Output ($/1M) ⇅</th>
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
    "updated_at": "2026-02-06",
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
        {"id": "gcp-chirp3-tts", "vendor": "google", "name": "GCP Chirp 3 (TTS)*", "input": 32.0, "output": 32.0}
    ]
};

// Track the currently displayed (filtered + sorted) data
let currentData = [...pricingData.prices];

function populateTable(data) {
    const body = document.getElementById('pricingBody');
    body.innerHTML = '';
    data.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="padding: 12px; border-bottom: 1px solid #eee;">${item.name}</td>
            <td style="padding: 12px; border-bottom: 1px solid #eee;">${item.vendor}</td>
            <td style="padding: 12px; border-bottom: 1px solid #eee;">$${item.input}</td>
            <td style="padding: 12px; border-bottom: 1px solid #eee;">$${item.output}</td>
        `;
        body.appendChild(row);
    });
}

function getModelType(item) {
    const n = item.name.toLowerCase();
    if (n.includes('image') || n.includes('canvas') || n.includes('imagen')) return 'image';
    if (n.includes('tts') || n.includes('polly')) return 'speech';
    if (n.includes('stt') || n.includes('transcribe')) return 'transcription';
    return 'text';
}

function getFilteredData() {
    const searchTerm = document.getElementById('pricingSearch').value.toLowerCase();
    return pricingData.prices.filter(item =>
        item.name.toLowerCase().includes(searchTerm) ||
        item.vendor.toLowerCase().includes(searchTerm) ||
        item.id.toLowerCase().includes(searchTerm) ||
        getModelType(item).includes(searchTerm)
    );
}

function filterTable() {
    currentData = getFilteredData();
    populateTable(currentData);
    renderChart(currentData);
}

let sortOrder = [1, 1, 1, 1];
function sortTable(columnIndex) {
    const propertyMap = ['name', 'vendor', 'input', 'output'];
    const property = propertyMap[columnIndex];

    currentData.sort((a, b) => {
        let valA = a[property];
        let valB = b[property];

        if (typeof valA === 'string') {
            return valA.localeCompare(valB) * sortOrder[columnIndex];
        } else {
            return (valA - valB) * sortOrder[columnIndex];
        }
    });

    sortOrder[columnIndex] *= -1;
    populateTable(currentData);
    renderChart(currentData);
}

// Chart.js Visualization
let chartInstance = null;

function renderChart(data) {
    const ctx = document.getElementById('pricingChart').getContext('2d');

    // Shapes: Google=Circle, AWS=Triangle, OpenAI=Rect, Anthropic=Diamond
    // Colors: Text=Blue, Image=Red, Transcription=Green, Speech=Purple

    const datasets = data.map(item => {
        // Color by vendor
        let color = '#4285F4'; // Google blue
        if (item.vendor === 'amazon') color = '#FF9900'; // Amazon orange
        else if (item.vendor === 'openai') color = '#10A37F'; // OpenAI green
        else if (item.vendor === 'anthropic') color = '#D97706'; // Anthropic amber

        // Shape by type
        let pointStyle = 'circle'; // Text default
        if (item.name.toLowerCase().includes('image') || item.name.toLowerCase().includes('canvas') || item.name.toLowerCase().includes('imagen')) {
            pointStyle = 'star';
        } else if (item.name.toLowerCase().includes('tts') || item.name.toLowerCase().includes('polly')) {
            pointStyle = 'rectRot'; // Diamond
        } else if (item.name.toLowerCase().includes('stt') || item.name.toLowerCase().includes('transcribe')) {
            pointStyle = 'triangle';
        }

        return {
            label: item.name,
            data: [{x: item.input, y: item.output}],
            backgroundColor: color,
            borderColor: color,
            pointStyle: pointStyle,
            pointRadius: 6,
            pointHoverRadius: 8
        };
    });

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': Input $' + context.raw.x + ', Output $' + context.raw.y;
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Model Pricing: Input vs Output Cost ($ per 1M tokens)'
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy'
                    },
                    zoom: {
                        wheel: {
                            enabled: true
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy'
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Input Cost ($)'
                    },
                    beginAtZero: true
                },
                y: {
                    title: {
                        display: true,
                        text: 'Output Cost ($)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

function buildLegend() {
    const el = document.getElementById('chartLegend');
    const vendorItems = [
        {color: '#4285F4', label: 'Google'},
        {color: '#FF9900', label: 'Amazon'},
        {color: '#10A37F', label: 'OpenAI'},
        {color: '#D97706', label: 'Anthropic'}
    ];
    const typeItems = [
        {symbol: '●', label: 'Text'},
        {symbol: '✦', label: 'Image'},
        {symbol: '▲', label: 'Transcription'},
        {symbol: '◆', label: 'Speech'}
    ];
    let html = '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:20px;margin-bottom:8px;font-size:14px;">';
    html += '<strong style="color:#666">Vendor:</strong> ';
    vendorItems.forEach(v => { html += `<span style="color:${v.color}">■ ${v.label}</span>`; });
    html += '</div>';
    html += '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:20px;margin-bottom:24px;font-size:14px;">';
    html += '<strong style="color:#666">Type:</strong> ';
    typeItems.forEach(t => { html += `<span>${t.symbol} ${t.label}</span>`; });
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
renderChart(pricingData.prices);
</script>
