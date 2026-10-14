// 50 Startups 特徵選擇比較：只讀取 results.json 呈現，不在前端重算模型
"use strict";

const LABELS = {
  "R&D Spend": "R&D Spend",
  "Administration": "Administration",
  "Marketing Spend": "Marketing Spend",
  "State_Florida": "State: Florida",
  "State_New York": "State: New York",
};
const SHORT = {
  "R&D Spend": "R&D",
  "Administration": "Admin",
  "Marketing Spend": "Marketing",
  "State_Florida": "FL",
  "State_New York": "NY",
};
const METHOD_COLORS = {
  lasso: "#2f6fdf",
  forward: "#e07b00",
  rfe: "#1d9a5b",
  elasticnet: "#c2389a",
  exhaustive: "#6b4fd8",
};
const METHOD_POINTS = {
  lasso: "circle",
  forward: "rect",
  rfe: "triangle",
  elasticnet: "rectRot",
  exhaustive: "star",
};
// Compare 模式下各方法在 x 軸的水平偏移，避免選到相同特徵的方法曲線完全重疊
const METHOD_OFFSETS = { lasso: -0.16, forward: -0.08, rfe: 0, elasticnet: 0.08, exhaustive: 0.16 };
const BENCHMARK = "exhaustive";

let DATA = null;
const state = { method: BENCHMARK, compare: false, k: 2 };
const charts = {};

// ---------- 格式化 ----------
const money = (v, d = 2) =>
  (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const num = (v, d = 4) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pval = (p) => (p < 1e-4 ? "< 0.0001" : p.toFixed(4));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const $ = (id) => document.getElementById(id);

const methodById = (id) => DATA.methods.find((m) => m.id === id);
const currentMethod = () => methodById(state.compare ? BENCHMARK : state.method);
const currentResult = () => currentMethod().results[String(state.k)];

// ---------- 頁首 ----------
function renderHeader() {
  const g = DATA.global_best;
  const m = DATA.meta;
  const methodsNames = g.methods.map((id) => methodById(id).name).join("、");
  $("badges").innerHTML = `
    <div class="badge"><div class="k">資料集</div><div class="v">${m.n_rows} 筆（Train ${m.n_train} / Test ${m.n_test}）</div></div>
    <div class="badge"><div class="k">比較方法</div><div class="v">5 種特徵選擇 × k = 1～5</div></div>
    <div class="badge best"><div class="k">全域最佳 k（依 CV RMSE）</div>
      <div class="v">k = ${g.k}：${g.features.map((f) => SHORT[f]).join(" + ")}</div></div>
    <div class="badge best"><div class="k">最佳 k 的 RMSE</div>
      <div class="v">CV ${money(g.cv_rmse_mean, 0)} ± ${money(g.cv_rmse_std, 0)}</div>
      <div class="k">Test ${money(g.test_rmse, 0)} · 選到此組合的方法：${esc(methodsNames)}</div></div>`;
  $("footer-meta").textContent =
    `分析產生時間：${m.generated_at}（UTC）· Python ${m.versions.python} · scikit-learn ${m.versions["scikit-learn"]} · ` +
    `statsmodels ${m.versions.statsmodels} · 切分：${m.split} · CV：${m.cv}`;
}

// ---------- 方法按鈕 ----------
function renderMethodButtons() {
  const html = DATA.methods.map((m) => {
    let matchTag = "";
    if (m.id === BENCHMARK) matchTag = `<span class="tag">Benchmark</span>`;
    else if (m.matches_benchmark_all) matchTag = `<span class="tag match">Matches Benchmark</span>`;
    else matchTag = `<span class="tag partial">${m.matches_benchmark_count}/5 Match</span>`;
    const pressed = !state.compare && state.method === m.id;
    return `<button class="btn" type="button" data-method="${m.id}" aria-pressed="${pressed}">
      <span class="title">${esc(m.name)}</span>
      <span class="tags"><span class="tag">${esc(m.tag)}</span>${matchTag}</span></button>`;
  }).join("");
  $("method-buttons").innerHTML = html + `
    <button class="btn compare" type="button" data-method="__compare" aria-pressed="${state.compare}">
      <span class="title">Compare All 5 Methods</span>
      <span class="tags"><span class="tag">疊圖比較</span></span></button>`;

  $("method-desc").textContent = state.compare
    ? "疊加 5 種方法的 CV 與 Test 曲線，並列出各方法在目前 k 選到的特徵。選到相同特徵的方法數值相同；為了辨識，各方法的點在水平方向稍微錯開（數值不變）。"
    : currentMethod().description;
}

// ---------- k 按鈕 ----------
function renderKButtons() {
  const m = currentMethod();
  $("k-buttons").innerHTML = [1, 2, 3, 4, 5].map((k) => {
    const r = m.results[String(k)];
    const best = k === m.best_k;
    return `<button class="btn" type="button" data-k="${k}" aria-pressed="${k === state.k}">
      <span class="title">k = ${k}${best ? ' <span class="star" title="最佳 k">★</span>' : ""}</span>
      <span class="k-feats">${r.features.map((f) => esc(LABELS[f])).join("、")}</span>
      <span class="k-meta">CV RMSE ${money(r.cv.rmse_mean, 0)}</span></button>`;
  }).join("");
}

// ---------- Compare 面板 ----------
function renderCompare() {
  $("compare-panel").hidden = !state.compare;
  if (!state.compare) return;
  $("compare-k").textContent = state.k;
  const rows = DATA.methods.map((m) => {
    const r = m.results[String(state.k)];
    const matchCell = m.id === BENCHMARK ? "—" : r.matches_benchmark ? '<span class="yes">✓</span>' : '<span class="no">✗</span>';
    return `<tr><td>${esc(m.name)}</td>
      <td class="left">${r.features.map((f) => esc(LABELS[f])).join("、")}</td>
      <td>${money(r.cv.rmse_mean)}</td><td>${money(r.test.rmse)}</td>
      <td>${num(r.test.r2)}</td><td>${num(r.train.adj_r2)}</td><td>${matchCell}</td>
      <td>${m.best_k}</td></tr>`;
  }).join("");
  $("compare-table").innerHTML = `<thead><tr><th>方法</th><th class="left">選到的特徵</th>
    <th>CV RMSE</th><th>Test RMSE</th><th>Test R²</th><th>Adj. R²（Train）</th><th>Matches Benchmark</th><th>最佳 k</th></tr></thead>
    <tbody>${rows}</tbody>`;

  const head = [1, 2, 3, 4, 5].map((k) => `<th class="left">k = ${k}</th>`).join("");
  const body = DATA.methods.map((m) => `<tr><td>${esc(m.name)}</td>${[1, 2, 3, 4, 5].map((k) => {
    const r = m.results[String(k)];
    const mark = m.id === BENCHMARK ? "" : r.matches_benchmark ? ' <span class="yes">✓</span>' : ' <span class="no">✗</span>';
    return `<td class="left">${r.features.map((f) => SHORT[f]).join(" + ")}${mark}</td>`;
  }).join("")}</tr>`).join("");
  $("feature-matrix").innerHTML = `<thead><tr><th>方法</th>${head}</tr></thead><tbody>${body}</tbody>`;
}

// ---------- Compare 模式：說明曲線為何重疊（依 results.json 實際結果產生） ----------
function renderOverlapNote() {
  const note = $("overlap-note");
  note.hidden = !state.compare;
  if (!state.compare) return;
  const bench = methodById(BENCHMARK);
  const others = DATA.methods.filter((m) => m.id !== BENCHMARK);
  const same = others.filter((m) => m.matches_benchmark_all).map((m) => m.name);
  const parts = [];
  if (same.length) {
    parts.push(`${same.join("、")} 在 k = 1～5 都與 ${bench.name} 選到相同特徵，數值完全相同，因此曲線重疊。`);
  }
  for (const m of others.filter((x) => !x.matches_benchmark_all)) {
    const diffs = [1, 2, 3, 4, 5].filter((k) => !m.results[String(k)].matches_benchmark).map((k) => {
      const mine = m.results[String(k)];
      const ref = bench.results[String(k)];
      const add = mine.features.filter((f) => !ref.features.includes(f)).map((f) => LABELS[f]);
      const drop = ref.features.filter((f) => !mine.features.includes(f)).map((f) => LABELS[f]);
      const delta = mine.cv.rmse_mean - ref.cv.rmse_mean;
      return `k = ${k}（選 ${add.join("、")}，而非 ${drop.join("、")}；CV RMSE 差 ${money(Math.abs(delta), 0)}）`;
    });
    parts.push(`${m.name} 只在 ${diffs.join("；")} 與 ${bench.name} 不同，其餘 k 也與其重疊。`);
  }
  parts.push("圖中各方法的點在水平方向稍微錯開，只是為了辨識，數值不變。");
  note.textContent = parts.join(" ");
}

// ---------- 指標卡片 ----------
function renderCards() {
  const r = currentResult();
  const card = (name, src, srcCls, value, sub) => `
    <div class="card"><div class="head"><span class="name">${name}</span><span class="src ${srcCls}">${src}</span></div>
    <div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  $("metric-cards").innerHTML =
    card("Test RMSE", "Test", "test", money(r.test.rmse), `CV RMSE ${money(r.cv.rmse_mean)} ± ${money(r.cv.rmse_std)}（Train CV）`) +
    card("Test R²", "Test", "test", num(r.test.r2), `CV R² ${num(r.cv.r2_mean)} ± ${num(r.cv.r2_std)}（Train CV）`) +
    card("Adjusted R²", "Train", "train", num(r.train.adj_r2), `R² ${num(r.train.r2)} · n = ${r.train.n}`) +
    card("Test MAE", "Test", "test", money(r.test.mae), `測試集 ${r.test.rows.length} 筆平均絕對誤差`) +
    card("F-statistic", "Train", "train", num(r.train.f_stat, 2), `p ${r.train.f_pvalue < 1e-4 ? "< 0.0001" : "= " + r.train.f_pvalue.toFixed(4)}`);
}

// ---------- Active Predictors ----------
function renderPredictors() {
  const r = currentResult();
  $("pred-count").textContent = `（${r.features.length}）`;
  const who = state.compare ? `<span class="hint">顯示 benchmark：Exhaustive Search</span>` : "";
  $("active-predictors").innerHTML = r.features.map((f) => `<span class="chip">${esc(LABELS[f])}</span>`).join("") + who;
}

// ---------- 圖表 ----------
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function buildDatasets(metric) {
  const ks = [1, 2, 3, 4, 5];
  const methods = state.compare ? DATA.methods : [currentMethod()];
  const sets = [];
  for (const m of methods) {
    const color = state.compare ? METHOD_COLORS[m.id] : cssVar("--accent");
    const testColor = state.compare ? METHOD_COLORS[m.id] : cssVar("--good");
    const pick = (k, src) => {
      const r = m.results[String(k)];
      if (metric === "rmse") return src === "cv" ? r.cv.rmse_mean : r.test.rmse;
      return src === "cv" ? r.cv.r2_mean : r.test.r2;
    };
    const radius = (ctx) => (ks[ctx.dataIndex] === state.k ? 7 : 4);
    const dx = state.compare ? METHOD_OFFSETS[m.id] : 0;
    const cvPoint = state.compare ? METHOD_POINTS[m.id] : "circle";
    sets.push({
      label: state.compare ? `${m.name} · CV` : "CV（Train）",
      data: ks.map((k) => ({ x: k + dx, y: pick(k, "cv") })),
      borderColor: color, backgroundColor: color, borderWidth: 2.5,
      pointRadius: radius, pointHoverRadius: 8, pointStyle: cvPoint, tension: 0.15,
      _std: ks.map((k) => (metric === "rmse" ? m.results[String(k)].cv.rmse_std : m.results[String(k)].cv.r2_std)),
    });
    sets.push({
      label: state.compare ? `${m.name} · Test` : "Test",
      data: ks.map((k) => ({ x: k + dx, y: pick(k, "test") })),
      borderColor: testColor, backgroundColor: testColor, borderWidth: 2, borderDash: [6, 4],
      pointRadius: radius, pointHoverRadius: 8, pointStyle: state.compare ? METHOD_POINTS[m.id] : "rectRot",
      tension: 0.15,
    });
  }
  return sets;
}

function renderCharts() {
  const text = cssVar("--muted");
  const grid = cssVar("--border");
  const make = (id, metric) => {
    const fmt = metric === "rmse" ? (v) => money(v, 0) : (v) => num(v, 3);
    const config = {
      type: "line",
      data: { datasets: buildDatasets(metric) },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { color: text, boxWidth: 14, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              title: (items) => `k = ${Math.round(items[0].parsed.x)}`,
              label: (ctx) => {
                const std = ctx.dataset._std ? ` ± ${fmt(ctx.dataset._std[ctx.dataIndex])}` : "";
                return `${ctx.dataset.label}: ${fmt(ctx.parsed.y)}${std}`;
              },
            },
          },
        },
        scales: {
          x: { type: "linear", min: 0.6, max: 5.4,
               ticks: { color: text, stepSize: 1, callback: (v) => (Number.isInteger(v) ? `k=${v}` : "") },
               grid: { color: grid } },
          y: { ticks: { color: text, callback: fmt }, grid: { color: grid },
               title: { display: true, text: metric === "rmse" ? "RMSE ($)" : "R²", color: text } },
        },
        onClick: (_e, els) => { if (els.length) setK(els[0].index + 1); },
      },
    };
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart($(id), config);
  };
  make("chart-rmse", "rmse");
  make("chart-r2", "r2");
}

// ---------- 方程式 ----------
function renderEquation() {
  const r = currentResult();
  const [b0, ...terms] = r.coefficients;
  const parts = terms.map((c) => {
    const sign = c.estimate < 0 ? "−" : "+";
    return ` ${sign} <span class="coef">${num(Math.abs(c.estimate), 4)}</span> × (${esc(LABELS[c.name])})`;
  }).join("");
  $("equation").innerHTML = `Profit = <span class="coef">${num(b0.estimate, 2)}</span>${parts}`;
}

// ---------- 係數表 ----------
function renderCoefTable() {
  const r = currentResult();
  const rows = r.coefficients.map((c) => {
    const sig = c.p_value < 0.05 ? "sig" : "nsig";
    return `<tr><td>${esc(LABELS[c.name] || c.name)}</td><td>${num(c.estimate)}</td><td>${num(c.std_error)}</td>
      <td>${num(c.t_stat, 3)}</td><td class="${sig}">${pval(c.p_value)}</td>
      <td>[${num(c.ci_low)}, ${num(c.ci_high)}]</td><td>${c.vif == null ? "—" : num(c.vif, 3)}</td></tr>`;
  }).join("");
  $("coef-table").innerHTML = `<thead><tr><th>Predictor</th><th>Estimate (β)</th><th>Std. Error</th><th>t-Stat</th>
    <th>p-Value</th><th>95% CI</th><th>VIF</th></tr></thead><tbody>${rows}</tbody>`;
}

// ---------- 即時預測器 ----------
function initPredictor() {
  const d = DATA.meta.input_defaults;
  $("in-rd").value = Math.round(d["R&D Spend"]);
  $("in-admin").value = Math.round(d["Administration"]);
  $("in-mkt").value = Math.round(d["Marketing Spend"]);
  $("predict-form").addEventListener("input", renderPrediction);
  $("predict-form").addEventListener("submit", (e) => e.preventDefault());
}

function renderPrediction() {
  const r = currentResult();
  const st = $("in-state").value;
  const x = {
    "R&D Spend": parseFloat($("in-rd").value) || 0,
    "Administration": parseFloat($("in-admin").value) || 0,
    "Marketing Spend": parseFloat($("in-mkt").value) || 0,
    "State_Florida": st === "Florida" ? 1 : 0,
    "State_New York": st === "New York" ? 1 : 0,
  };
  const used = new Set(r.features);
  const inputs = { "in-rd": "R&D Spend", "in-admin": "Administration", "in-mkt": "Marketing Spend" };
  for (const [id, f] of Object.entries(inputs)) $(id).parentElement.classList.toggle("unused", !used.has(f));
  const usesState = used.has("State_Florida") || used.has("State_New York");
  $("in-state").parentElement.classList.toggle("unused", !usesState);

  const [b0, ...terms] = r.coefficients;
  let y = b0.estimate;
  let formula = `${num(b0.estimate, 2)}`;
  for (const c of terms) {
    y += c.estimate * x[c.name];
    const sign = c.estimate < 0 ? " − " : " + ";
    const xv = c.name.startsWith("State_") ? x[c.name] : x[c.name].toLocaleString("en-US");
    formula += `${sign}${num(Math.abs(c.estimate), 4)} × ${xv}`;
  }
  $("pred-value").textContent = money(y);
  $("pred-formula").textContent = `Profit = ${formula} = ${money(y)}`;
  const model = state.compare ? "Exhaustive Search（benchmark）" : currentMethod().name;
  $("pred-note").textContent = `使用模型：${model}，k = ${state.k}（訓練集 OLS 係數）。`;
}

// ---------- 殘差診斷 ----------
function renderResiduals() {
  const r = currentResult();
  const rows = [...r.test.rows].sort((a, b) => Math.abs(b.residual) - Math.abs(a.residual));
  const status = (p) => {
    const a = Math.abs(p);
    if (a < 5) return '<span class="status ok">準確</span>';
    if (a < 15) return '<span class="status mid">尚可</span>';
    return '<span class="status hi">偏差大</span>';
  };
  $("resid-table").innerHTML = `<thead><tr><th>Rank</th><th>Sample ID</th><th class="left">State</th><th>實際值</th>
    <th>預測值</th><th>殘差</th><th>|殘差|</th><th>% 誤差</th><th class="left">狀態</th></tr></thead><tbody>${
    rows.map((x, i) => `<tr><td>${i + 1}</td><td>#${x.sample_id}</td><td class="left">${esc(x.state)}</td>
      <td>${money(x.actual)}</td><td>${money(x.predicted)}</td><td>${money(x.residual)}</td>
      <td>${money(Math.abs(x.residual))}</td><td>${num(x.pct_error, 2)}%</td><td class="left">${status(x.pct_error)}</td></tr>`).join("")
  }</tbody>`;
}

// ---------- 狀態更新 ----------
function renderAll() {
  renderMethodButtons();
  renderKButtons();
  renderCompare();
  renderOverlapNote();
  renderCards();
  renderPredictors();
  renderCharts();
  renderEquation();
  renderCoefTable();
  renderPrediction();
  renderResiduals();
}

function setK(k) {
  state.k = k;
  renderAll();
}

function bindEvents() {
  $("method-buttons").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-method]");
    if (!btn) return;
    if (btn.dataset.method === "__compare") state.compare = true;
    else { state.compare = false; state.method = btn.dataset.method; }
    renderAll();
  });
  $("k-buttons").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-k]");
    if (btn) setK(Number(btn.dataset.k));
  });
  // 系統深淺色切換時重畫圖表
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderCharts);
}

async function init() {
  try {
    const res = await fetch("results.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    DATA = await res.json();
  } catch (err) {
    $("load-error").hidden = false;
    console.error(err);
    return;
  }
  state.method = BENCHMARK;
  state.k = DATA.global_best.k;
  $("app").hidden = false;
  renderHeader();
  initPredictor();
  bindEvents();
  renderAll();
}

init();
