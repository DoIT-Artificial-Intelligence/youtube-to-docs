// Pricing page: cost-per-video bars and a sortable, filterable price list.
// The data is injected at build time from youtube_to_docs/prices.py
// (scripts/docs/pricing_hook.py).
(function () {
  const dataEl = document.getElementById("pricing-data");
  if (!dataEl) return;

  const { prices, assumptions } = JSON.parse(dataEl.textContent);
  const VENDORS = { google: "Google", amazon: "Amazon", anthropic: "Anthropic", openai: "OpenAI" };

  const state = { minutes: 60, vendor: null, type: "text", search: "", sort: null };

  const money = (v) => {
    if (v === null || v === undefined) return "—";
    if (v === 0) return "Free";
    if (v < 0.01) return "$" + +v.toPrecision(2);
    if (v < 1) return "$" + v.toFixed(v < 0.1 ? 3 : 2).replace(/0$/, "");
    if (v < 100) return "$" + v.toFixed(2).replace(/\.00$/, "");
    return "$" + Math.round(v);
  };
  const rate = (v) => (v === null || v === undefined ? "—" : v === 0 ? "Free" : "$" + +v.toFixed(4));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const vendorLabel = (v) => VENDORS[v] || v;

  // Cost of one summary pass over a transcript of `minutes` of speech.
  const passCost = (p, minutes) =>
    (minutes * assumptions.tokens_per_minute * p.input + assumptions.summary_output_tokens * p.output) / 1e6;

  const byVendor = (p) => !state.vendor || p.vendor === state.vendor;
  const textModels = () => prices.filter((p) => p.type === "text");

  // ---------- KPIs ----------
  function renderKpis() {
    // The CLI's default model, so the headline matches what a bare run costs.
    const flashLite = prices.find((p) => p.id === "gemini-3.5-flash-lite");
    const cost = passCost(flashLite, 60);
    const free = textModels().find((p) => p.input === 0 && p.output === 0);
    document.getElementById("kpi-cheapest").textContent = cost < 0.01 ? "under 1¢" : money(cost);
    document.getElementById("kpi-cheapest-model").textContent =
      `${money(cost)} with ${flashLite.name}, the default model` + (free ? `. ${free.name.replace(/ \(.*\)$/, "")} is free.` : ".");

    const stt = prices.filter((p) => p.type === "stt").sort((a, b) => a.per_hour - b.per_hour)[0];
    document.getElementById("kpi-stt").textContent = "from " + money(stt.per_hour);
    document.getElementById("kpi-stt-model").textContent = "with " + stt.name;
  }

  // ---------- Vendor chips ----------
  function renderChips() {
    const el = document.getElementById("pricing-vendors");
    const opts = [[null, "All vendors"], ...Object.entries(VENDORS)];
    el.innerHTML = opts
      .map(([v, label]) => `<button type="button" class="pricing-chip" data-vendor="${v || ""}" aria-pressed="${state.vendor === v}">${label}</button>`)
      .join("");
  }
  document.getElementById("pricing-vendors").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    state.vendor = btn.dataset.vendor || null;
    renderChips();
    renderBars();
    renderTabs();
    renderTable();
  });

  // ---------- Bars ----------
  const tooltip = document.getElementById("pricing-tooltip");

  function renderBars() {
    const rows = textModels()
      .filter(byVendor)
      .map((p) => ({ p, cost: passCost(p, state.minutes) }))
      .sort((a, b) => a.cost - b.cost);
    const max = Math.max(...rows.map((r) => r.cost), 1e-9);
    document.getElementById("pricing-bars").innerHTML = rows
      .map(
        ({ p, cost }) => `
        <div class="pricing-bar" role="listitem" tabindex="0" data-id="${esc(p.id)}"
             aria-label="${esc(p.name)}, ${vendorLabel(p.vendor)}: ${money(cost)}">
          <span class="pricing-bar__label">${esc(p.name)}<small>${vendorLabel(p.vendor)}</small></span>
          <span class="pricing-bar__track"><span class="pricing-bar__fill" style="width:${Math.max((cost / max) * 100, 0.6)}%"></span></span>
          <span class="pricing-bar__value">${money(cost)}</span>
        </div>`
      )
      .join("");
  }

  function showTooltip(row, x, y) {
    const p = prices.find((q) => q.id === row.dataset.id);
    const tokensIn = state.minutes * assumptions.tokens_per_minute;
    tooltip.innerHTML = `<strong>${esc(p.name)}</strong>
      <span>Input: ${tokensIn.toLocaleString()} tokens × ${rate(p.input)}/1M</span>
      <span>Output: ${assumptions.summary_output_tokens.toLocaleString()} tokens × ${rate(p.output)}/1M</span>
      ${p.note ? `<em>${esc(p.note)}</em>` : ""}`;
    tooltip.hidden = false;
    const box = tooltip.getBoundingClientRect();
    tooltip.style.left = Math.min(x + 14, window.innerWidth - box.width - 8) + "px";
    tooltip.style.top = y + 14 + "px";
  }
  const bars = document.getElementById("pricing-bars");
  bars.addEventListener("mousemove", (e) => {
    const row = e.target.closest(".pricing-bar");
    row ? showTooltip(row, e.clientX, e.clientY) : (tooltip.hidden = true);
  });
  bars.addEventListener("mouseleave", () => (tooltip.hidden = true));
  bars.addEventListener("focusin", (e) => {
    const row = e.target.closest(".pricing-bar");
    if (!row) return;
    const r = row.getBoundingClientRect();
    showTooltip(row, r.left + r.width / 2, r.bottom - 8);
  });
  bars.addEventListener("focusout", () => (tooltip.hidden = true));
  window.addEventListener("scroll", () => (tooltip.hidden = true), { passive: true });

  const slider = document.getElementById("pricing-minutes");
  slider.addEventListener("input", () => {
    state.minutes = +slider.value;
    const h = Math.floor(state.minutes / 60), m = state.minutes % 60;
    document.getElementById("pricing-minutes-label").textContent =
      [h ? `${h} hour${h > 1 ? "s" : ""}` : "", m ? `${m} min` : ""].filter(Boolean).join(" ");
    renderBars();
  });

  // ---------- Price list ----------
  const name = { key: "name", label: "Model", fmt: (p) => `${esc(p.name)}${p.note ? ` <span class="pricing-note" title="${esc(p.note)}">${p.note_label}</span>` : ""}` };
  const vendor = { key: "vendor", label: "Vendor", fmt: (p) => vendorLabel(p.vendor) };
  const tokIn = { key: "input", label: "Input $/1M", num: true, fmt: (p) => rate(p.input) };
  const tokOut = { key: "output", label: "Output $/1M", num: true, fmt: (p) => rate(p.output) };
  const unit = { key: "unit_price", label: "Price", num: true, fmt: (p) => (p.unit ? `${rate(p.unit_price)} <small>${p.unit}</small>` : `${rate(p.input)} / ${rate(p.output)} <small>per 1M tokens</small>`) };

  const TABS = {
    text: {
      label: "Text",
      cols: [name, vendor, tokIn,
        { key: "cached", label: "Cached input $/1M", num: true, fmt: (p) => rate(p.cached) },
        tokOut,
        { key: "hour", label: "1-hour summary", num: true, value: (p) => passCost(p, 60), fmt: (p) => money(passCost(p, 60)) }],
    },
    image: {
      label: "Image",
      cols: [name, vendor,
        { ...tokIn, fmt: (p) => rate(p.input) + (p.converted ? "*" : "") },
        { ...tokOut, fmt: (p) => rate(p.output) + (p.converted ? "*" : "") },
        { key: "measured", label: "Measured per infographic", num: true, fmt: (p) => money(p.measured) }],
    },
    stt: {
      label: "Speech-to-text",
      cols: [name, vendor, unit, { key: "per_hour", label: "≈ per hour of audio", num: true, fmt: (p) => money(p.per_hour) }],
    },
    tts: {
      label: "Text-to-speech",
      cols: [name, vendor, unit, { key: "per_hour", label: "≈ per hour of speech", num: true, fmt: (p) => money(p.per_hour) }],
    },
    translate: {
      label: "Translation",
      cols: [name, vendor, unit, { key: "per_hour", label: "≈ per hour of transcript", num: true, fmt: (p) => money(p.per_hour) }],
      footnote: "LLMs can translate too: pass any text model, e.g. --translate gemini-3.5-flash-lite-es.",
    },
  };

  function renderTabs() {
    document.getElementById("pricing-tabs").innerHTML = Object.entries(TABS)
      .map(([k, t]) => {
        // Counts follow the search and vendor filter, so matches in other tabs are visible.
        const n = prices.filter((p) => p.type === k && matches(p)).length;
        return `<button type="button" role="tab" class="pricing-tab" data-type="${k}" aria-selected="${state.type === k}">${t.label} <span>${n}</span></button>`;
      })
      .join("");
  }
  document.getElementById("pricing-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    state.type = btn.dataset.type;
    state.sort = null;
    renderTabs();
    renderTable();
  });

  const matches = (p) => {
    const q = state.search.trim().toLowerCase().replace(/[-_]/g, " ");
    return byVendor(p) && (!q || `${p.name} ${p.id} ${p.vendor}`.toLowerCase().replace(/[-_]/g, " ").includes(q));
  };

  function renderTable() {
    const tab = TABS[state.type];
    let rows = prices.filter((p) => p.type === state.type && matches(p));
    if (state.sort) {
      const col = tab.cols.find((c) => c.key === state.sort.key);
      const val = col.value || ((p) => (col.num ? p[col.key] ?? Infinity : String(p[col.key])));
      rows = [...rows].sort((a, b) => {
        const x = val(a), y = val(b);
        return (typeof x === "string" ? x.localeCompare(y) : x - y) * state.sort.dir;
      });
    } else if (state.type === "text") {
      rows = [...rows].sort((a, b) => passCost(a, 60) - passCost(b, 60));
    }

    const head = tab.cols
      .map((c) => {
        const active = state.sort && state.sort.key === c.key;
        const ariaSort = active ? (state.sort.dir === 1 ? "ascending" : "descending") : "none";
        return `<th scope="col" aria-sort="${ariaSort}" class="${c.num ? "num" : ""}"><button type="button" data-key="${c.key}">${c.label}<span aria-hidden="true">${active ? (state.sort.dir === 1 ? " ↑" : " ↓") : ""}</span></button></th>`;
      })
      .join("");
    const body = rows.length
      ? rows.map((p) => `<tr>${tab.cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.fmt(p)}</td>`).join("")}</tr>`).join("")
      : `<tr><td colspan="${tab.cols.length}" class="pricing-empty">No models match.</td></tr>`;
    document.getElementById("pricing-table").innerHTML =
      `<caption class="pricing-sr">${tab.label} model prices</caption><thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;

    const total = prices.filter((p) => p.type === state.type).length;
    document.getElementById("pricing-count").textContent =
      (rows.length === total ? `${total} models` : `${rows.length} of ${total} models`) + (tab.footnote ? ` · ${tab.footnote}` : "");
  }
  document.getElementById("pricing-table").addEventListener("click", (e) => {
    const btn = e.target.closest("th button");
    if (!btn) return;
    const key = btn.dataset.key;
    state.sort = state.sort && state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: 1 };
    renderTable();
  });
  document.getElementById("pricing-search").addEventListener("input", (e) => {
    state.search = e.target.value;
    // Jump to the first tab with matches if the current one has none.
    if (!prices.some((p) => p.type === state.type && matches(p))) {
      const hit = Object.keys(TABS).find((k) => prices.some((p) => p.type === k && matches(p)));
      if (hit) state.type = hit;
    }
    renderTabs();
    renderTable();
  });

  renderKpis();
  renderChips();
  renderBars();
  renderTabs();
  renderTable();
})();
