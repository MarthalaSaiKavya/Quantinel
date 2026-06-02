/* ============================================================================
   QUANTINEL PM DASHBOARD — Frontend Logic
   Agentic Bloomberg Terminal
   ============================================================================ */

let DATA = null;
let grid = null;

document.addEventListener('DOMContentLoaded', () => {
    // Init GridStack
    grid = GridStack.init({
        cellHeight: 80,
        margin: 8,
        disableOneColumnMode: true,
        float: true,
    });

    initTabs();
    updateClock();
    setInterval(updateClock, 1000);
    fetchPipeline();

    document.getElementById('btn-run-pipeline').addEventListener('click', async (e) => {
        const btn = e.target;
        btn.textContent = 'EXECUTING...';
        btn.style.background = 'var(--red)';
        try {
            const res = await fetch('/api/pipeline/refresh');
            DATA = await res.json();
            renderAll();
        } catch (e) {
            console.error('Pipeline refresh failed:', e);
        }
        btn.textContent = 'EXECUTE RUN';
        btn.style.background = '';
    });
});

async function fetchPipeline() {
    try {
        const res = await fetch('/api/pipeline');
        DATA = await res.json();
        renderAll();
        setStatus('status-api', 'online');
    } catch (e) {
        console.error('Pipeline fetch failed:', e);
        setStatus('status-api', 'offline');
    }
}

function setStatus(id, state) {
    const dot = document.getElementById(id);
    if (!dot) return;
    dot.className = 'status-dot' + (state === 'offline' ? ' offline' : state === 'warning' ? ' warning' : '');
}

function updateClock() {
    const el = document.getElementById('topbar-time');
    if (el) el.textContent = new Date().toISOString().slice(11, 19) + ' UTC';
}

function fmt(v, dp = 2) { return v == null ? '—' : Number(v).toFixed(dp); }
function pct(v, dp = 2) { return v == null ? '—' : (v * 100).toFixed(dp) + '%'; }
function numClass(v) { return v > 0.0001 ? 'num-pos' : v < -0.0001 ? 'num-neg' : 'num-zero'; }

function stanceBadge(s) {
    const cls = s === 'LONG' ? 'badge-long' : s === 'SHORT' ? 'badge-short' : 'badge-flat';
    return `<span class="badge ${cls}">${s}</span>`;
}

function renderAll() {
    if (!DATA) return;
    renderDecision();
    renderNews();
    renderRisk();
    renderSizing();
    renderTrace();
    renderPMView();
}

function activeBranch() {
    const key = DATA.active_branch || 'normal';
    return DATA[key] || DATA.normal;
}

function renderDecision() {
    const d = DATA.decision;
    const branch = activeBranch();
    const el = document.getElementById('decision-content');

    let stances = '';
    for (const t of DATA.tickers) {
        const stance = d.stances[t] || 'FLAT';
        const rawW = branch.records[branch.records.length-1].weights[t] || 0;
        stances += `
            <div class="stance-card">
                <span class="t">${t}</span>
                ${stanceBadge(stance)}
                <span class="mono ${numClass(rawW)}">${pct(rawW, 1)}</span>
            </div>`;
    }

    el.innerHTML = `
        <div class="decision-hero">
            <div class="decision-winner-box">
                <div>
                    <div class="dim" style="font-size:0.75rem">RECOMMENDED PATH</div>
                    <div class="val">${d.winner}</div>
                </div>
                <div style="text-align:right">
                    <div class="dim" style="font-size:0.75rem">CONFIDENCE</div>
                    <div class="mono" style="font-size:1.1rem; color:var(--text-white)">${d.confidence}</div>
                </div>
            </div>
            <div class="reasoning-box">${d.rationale}</div>
            <div class="dim" style="font-size:0.75rem; margin-top:8px">FORWARD STANCES</div>
            <div class="stances-grid">${stances}</div>
            <div class="flex-between" style="border-top:1px solid var(--border); padding-top:8px; margin-top:8px">
                <span class="dim" style="font-size:0.75rem">GROSS EXPOSURE: <span class="mono" style="color:var(--text-white)">${d.gross_exposure}x</span></span>
                <span class="dim" style="font-size:0.75rem">RISK AVERSION: <span class="mono" style="color:var(--text-white)">${d.risk_aversion}</span></span>
            </div>
        </div>`;
}

function renderNews() {
    const intel = DATA.intelligence;
    const el = document.getElementById('news-content');
    
    let feed = '';
    const tickers = DATA.tickers;
    for (const t of tickers) {
        const heads = intel.headlines[t] || [];
        const sent = intel.sentiment[t] || 0;
        if (heads.length > 0) {
            feed += `
                <div class="news-item">
                    <span class="t">${t}</span>
                    <span class="h">${heads[0]}</span>
                    <span class="mono ${numClass(sent)}">${sent > 0 ? '+' : ''}${sent.toFixed(2)}</span>
                </div>`;
        }
    }

    el.innerHTML = `
        <div class="flex-col">
            <div class="flex-row">
                <span class="dim" style="font-size:0.75rem">KEY THEMES:</span>
                <span style="color:var(--amber); font-weight:700; font-size:0.8rem">${intel.themes.join(' / ').toUpperCase()}</span>
            </div>
            <div class="news-feed" style="margin-top:10px">${feed}</div>
        </div>`;
}

function renderRisk() {
    const r = DATA.risk;
    const el = document.getElementById('risk-content');

    let varRows = '';
    for (const t of DATA.tickers) {
        const v95 = r.var_95[t] || 0;
        const cv95 = r.cvar_95[t] || 0;
        varRows += `<tr><td>${t}</td><td class="right num-neg">${pct(v95)}</td><td class="right num-neg">${pct(cv95)}</td></tr>`;
    }

    let saRows = '';
    for (const m of r.models) {
        const rate = r.breach_rates[m] || 0;
        const pass = rate <= 0.08;
        saRows += `<tr><td>${m.toUpperCase()}</td><td class="right ${pass ? 'num-pos' : 'num-neg'}">${pct(rate, 1)}</td><td class="right">${pass ? '<span class="badge badge-long">PASS</span>' : '<span class="badge badge-short">FAIL</span>'}</td></tr>`;
    }

    el.innerHTML = `
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px">
            <div>
                <div class="flex-between" style="border-bottom:1px solid var(--border); padding-bottom:4px; margin-bottom:8px">
                    <span class="dim" style="font-size:0.75rem">TAIL RISK METRICS</span>
                    <span class="dim" style="font-size:0.75rem">DISAGREEMENT: <span class="mono" style="color:var(--amber)">${r.disagreement.toFixed(3)}</span></span>
                </div>
                <table class="data-table">
                    <thead><tr><th>TICKER</th><th class="right">VAR 95</th><th class="right">CVAR 95</th></tr></thead>
                    <tbody>${varRows}</tbody>
                </table>
            </div>
            <div>
                <div class="dim" style="font-size:0.75rem; border-bottom:1px solid var(--border); padding-bottom:4px; margin-bottom:8px">CALIBRATION AUDIT</div>
                <table class="data-table">
                    <thead><tr><th>MODEL</th><th class="right">BREACH RATE</th><th class="right">STATUS</th></tr></thead>
                    <tbody>${saRows}</tbody>
                </table>
            </div>
        </div>`;
}

function renderSizing() {
    const r = DATA.risk;
    const branch = activeBranch();
    const el = document.getElementById('sizing-content');

    const shrink = 1 / (1 + r.disagreement);
    let rows = '';
    
    for (const t of DATA.tickers) {
        const raw = branch.records[branch.records.length-1].weights[t] || 0;
        const finalW = raw * shrink;
        rows += `
            <tr>
                <td style="font-weight:700">${t}</td>
                <td class="right ${numClass(raw)}">${pct(raw, 1)}</td>
                <td class="right dim">x${shrink.toFixed(2)}</td>
                <td class="right ${numClass(finalW)}" style="font-weight:700">${pct(finalW, 1)}</td>
            </tr>`;
    }

    el.innerHTML = `
        <table class="data-table">
            <thead><tr><th>TICKER</th><th class="right">RAW</th><th class="right">SHRINK</th><th class="right">FINAL</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}

function renderTrace() {
    const trace = DATA.decision.decision_trace;
    const elTrace = document.getElementById('trace-content');
    let lis = trace.map(t => `<li style="margin-bottom:6px; font-family:var(--font-sans); font-size:0.85rem">:: ${t}</li>`).join('');
    elTrace.innerHTML = `<ul style="list-style:none; color:var(--text-main); padding-left:0">${lis}</ul>`;

    const logs = DATA.execution_logs || [];
    const elLogs = document.getElementById('logs-content');
    elLogs.innerHTML = logs.map(l => `<div>${l}</div>`).join('');
}

function renderPMView() {
    const elEquity = document.getElementById('equity-content');
    const elComp = document.getElementById('comparison-content');
    const branchLabel = (DATA.active_branch || 'normal').toUpperCase();
    
    // Comparison
    const n = DATA.normal.scorecard;
    const q = DATA.quantum.scorecard;
    elComp.innerHTML = `
        <table class="data-table">
            <thead><tr><th>METRIC</th><th class="right">NORMAL</th><th class="right">QUANTUM</th></tr></thead>
            <tbody>
                <tr><td>SHARPE</td><td class="right">${n.sharpe.toFixed(2)}</td><td class="right">${q.sharpe.toFixed(2)}</td></tr>
                <tr><td>RETURN</td><td class="right">${pct(n.total_return)}</td><td class="right">${pct(q.total_return)}</td></tr>
                <tr><td>DIR ACC</td><td class="right">${pct(n.directional_accuracy, 1)}</td><td class="right">${pct(q.directional_accuracy, 1)}</td></tr>
            </tbody>
        </table>
        <div class="dim" style="font-size:0.75rem; margin-top:8px">ACTIVE BRANCH: ${branchLabel}</div>`;

    // Equity curve via simple SVG (winner branch)
    const branch = activeBranch();
    const eq = branch.scorecard.equity_curve.values;
    const minE = Math.min(...eq), maxE = Math.max(...eq);
    const range = maxE - minE || 1;
    const w = 400, h = 150;
    const pts = eq.map((v, i) => `${i===0?'M':'L'}${(i/(eq.length-1)*w).toFixed(1)},${(h - ((v-minE)/range)*h).toFixed(1)}`).join(' ');
    
    elEquity.innerHTML = `
        <div style="margin-bottom:8px">
            <span class="dim" style="font-size:0.75rem">${branchLabel} SHARPE: </span><span style="color:var(--green); font-weight:700">${branch.scorecard.sharpe.toFixed(2)}</span>
            <span class="dim" style="font-size:0.75rem; margin-left:12px">RETURN: </span><span style="color:var(--green); font-weight:700">${pct(branch.scorecard.total_return)}</span>
        </div>
        <svg width="100%" height="150" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="border:1px solid var(--border)">
            <path d="${pts}" fill="none" stroke="var(--blue)" stroke-width="2"/>
        </svg>`;
}

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            const panel = btn.closest('.panel');
            panel.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            panel.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');
        });
    });
}
