/* ============================================================
   SENTIENT — Road Infrastructure Command Center
   Real basemap (CARTO dark + OSM labels) with risk overlays,
   a narrative time machine, and a budget planner.
   ============================================================ */
"use strict";

/* ---------- palette (kept in sync with styles.css) ---------- */
const TIER_COLOR = { High: "#e5484d", Medium: "#d99a2b", Low: "#5f6b76" };
const PLAN_COLOR = "#5b8dbe";
const ACCENT = "#c9a227";

const COST_PER_KM_CR = 0.65;    // preventive resurfacing, ₹ crore per km
const REACTIVE_MULT = 3.2;      // reactive repair cost multiplier
const FAIL_LIKELIHOOD = 0.55;   // modelled 3-yr failure likelihood in critical band
const PLAY_MS_PER_MONTH = 380;  // time machine pace — slow enough to read the story

/* ---------- state ---------- */
const S = {
  data: null,
  city: null,
  mode: "overview",
  roads: [],
  monthIdx: 0,
  playTimer: null,
  selected: null,
  hovered: null,          // numeric feature id
  plan: new Set(),
  budgetPicks: [],
  budgetCr: 25,
  mapReady: false,
};

/* ---------- dom ---------- */
const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");

/* ---------- helpers ---------- */
const fmt = (n, d = 0) => Number(n).toLocaleString("en-IN", { maximumFractionDigits: d });
const fmtCr = (n) => n >= 100 ? `₹${fmt(n)} cr` : `₹${fmt(n, n < 10 ? 1 : 0)} cr`;
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => 1 - Math.pow(1 - t, 3);
const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const monthLabel = (m) => m ? MONTH_NAMES[+m.split("-")[1] - 1] + " " + m.split("-")[0] : "—";
const monthNum = (m) => +m.split("-")[1];
const isMonsoon = (m) => { const n = monthNum(m); return n >= 6 && n <= 9; };

function cuts() { return S.data.city_cuts[S.city]; }

// Band classification for a raw tile-risk value against the city's cuts.
function bandOf(v) {
  const c = cuts();
  if (v == null) return null;
  if (v >= c.high) return "High";
  if (v >= c.medium) return "Medium";
  return "Low";
}

// Last known value in a series at or before index i.
function valueAt(series, i) {
  for (let j = Math.min(i, series.length - 1); j >= 0 && j > i - 6; j--) {
    if (series[j] != null) return series[j];
  }
  return null;
}

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2600);
}

function countUp(el, target, opts = {}) {
  const { prefix = "", suffix = "", decimals = 0, dur = 800 } = opts;
  const start = performance.now();
  const from = el._val || 0;
  el._val = target;
  function tick(now) {
    const t = Math.min(1, (now - start) / dur);
    el.textContent = prefix + fmt(lerp(from, target, ease(t)), decimals) + suffix;
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  clearTimeout(el._ct);
  el._ct = setTimeout(() => { el.textContent = prefix + fmt(target, decimals) + suffix; }, dur + 80);
}

// City-relative priority score (0–100), computed against all analyzed roads in the pipeline.
function score100(r) {
  return r.priority != null ? r.priority : Math.round(r.risk_pct);
}

function zoneName(z) {
  return { NE: "north-east", NW: "north-west", SE: "south-east", SW: "south-west" }[z] || z || "—";
}

function zoneOfTile(tileId) {
  try {
    const [, rest] = tileId.split("__");
    const [, sy, sx] = rest.split("_");
    return (+sy < 2 ? "N" : "S") + (+sx < 2 ? "W" : "E");
  } catch { return null; }
}

/* ============================================================
   MAP
   ============================================================ */
let map = null;

function initMap() {
  map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      sources: {
        carto: {
          type: "raster",
          tiles: ["a", "b", "c", "d"].map((s) => `https://${s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png`),
          tileSize: 256,
          attribution: "© OpenStreetMap contributors © CARTO",
        },
      },
      layers: [{ id: "base", type: "raster", source: "carto" }],
    },
    center: [77.6, 12.95],
    zoom: 10.4,
    minZoom: 8.5,
    maxZoom: 17,
    preserveDrawingBuffer: true, // enables map image export
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

  map.on("load", () => {
    map.addSource("heat", { type: "geojson", data: emptyFC() });
    map.addLayer({
      id: "heat",
      type: "circle",
      source: "heat",
      layout: { visibility: "none" },
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 34, 11, 110, 13, 320],
        "circle-blur": 1,
        "circle-color": [
          "interpolate", ["linear"], ["coalesce", ["feature-state", "v"], 0],
          0, "rgba(217,154,43,0)",
          0.5, "rgba(217,154,43,0.16)",
          1, "rgba(229,72,77,0.30)",
        ],
      },
    });

    map.addSource("roads", { type: "geojson", data: emptyFC() });
    // invisible wide hit-target for comfortable hover/click
    map.addLayer({
      id: "roads-hit",
      type: "line",
      source: "roads",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-width": 16, "line-color": "#000", "line-opacity": 0.001 },
    });
    map.addLayer({
      id: "roads",
      type: "line",
      source: "roads",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": ["to-color", ["coalesce", ["feature-state", "color"], TIER_COLOR.Low]],
        "line-opacity": ["coalesce", ["feature-state", "op"], 0.85],
        "line-width": [
          "interpolate", ["linear"], ["zoom"],
          10, ["case", ["boolean", ["feature-state", "hover"], false], 4,
            ["match", ["get", "tier"], "High", 2.6, "Medium", 2, 1.4]],
          14, ["case", ["boolean", ["feature-state", "hover"], false], 9,
            ["match", ["get", "tier"], "High", 6.5, "Medium", 5, 3.5]],
        ],
      },
    });

    map.addSource("selected", { type: "geojson", data: emptyFC() });
    map.addLayer({
      id: "selected-casing",
      type: "line",
      source: "selected",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ffffff", "line-width": ["interpolate", ["linear"], ["zoom"], 10, 7, 14, 14], "line-opacity": 0.35 },
    });
    map.addLayer({
      id: "selected-line",
      type: "line",
      source: "selected",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ffffff", "line-width": ["interpolate", ["linear"], ["zoom"], 10, 3, 14, 6], "line-opacity": 0.95 },
    });

    map.on("mousemove", "roads-hit", onRoadHover);
    map.on("mouseleave", "roads-hit", clearHover);
    map.on("click", "roads-hit", (e) => {
      const f = e.features && e.features[0];
      if (f) selectRoad(S.roads[f.id]);
    });

    S.mapReady = true;
    if (S.data) loadCityIntoMap(false);
    $("splash").classList.add("gone");
  });
}

const emptyFC = () => ({ type: "FeatureCollection", features: [] });

function roadsFC() {
  return {
    type: "FeatureCollection",
    features: S.roads.map((r, i) => ({
      type: "Feature",
      id: i,
      properties: { tier: r.risk_level },
      geometry: { type: "LineString", coordinates: r.path },
    })),
  };
}

function heatFC() {
  return {
    type: "FeatureCollection",
    features: S.data.cities[S.city].tiles.map((t, i) => ({
      type: "Feature",
      id: i,
      properties: {},
      geometry: { type: "Point", coordinates: [t.lon, t.lat] },
    })),
  };
}

function fitCity(animate = true) {
  const [w, s, e, n] = S.data.cities[S.city].bbox;
  map.fitBounds([[w, s], [e, n]], { padding: { top: 40, bottom: 90, left: 360, right: 60 }, duration: animate ? 900 : 0 });
}

function loadCityIntoMap(animate = true) {
  map.removeFeatureState({ source: "roads" });
  map.removeFeatureState({ source: "heat" });
  map.getSource("roads").setData(roadsFC());
  map.getSource("heat").setData(heatFC());
  map.getSource("selected").setData(emptyFC());
  applyRoadStates();
  applyHeatStates();
  fitCity(animate);
}

/* Recolor every road according to the active mode. */
function applyRoadStates() {
  if (!S.mapReady) return;
  const picked = S.mode === "budget" ? new Set(S.budgetPicks.map((r) => r.id)) : null;

  S.roads.forEach((r, i) => {
    let color, op;
    if (S.mode === "budget") {
      if (picked.has(r.id)) { color = PLAN_COLOR; op = 0.95; }
      else { color = TIER_COLOR[r.risk_level]; op = 0.15; }
    } else if (S.mode === "time") {
      const v = valueAt(r.trend, Math.round(S.monthIdx));
      const band = bandOf(v);
      if (band == null) { color = "#4a545f"; op = 0.25; }
      else if (band === "Low") { color = "#6d7a87"; op = 0.55; }
      else { color = TIER_COLOR[band]; op = 0.95; }
    } else {
      color = S.plan.has(r.id) ? PLAN_COLOR : TIER_COLOR[r.risk_level];
      op = r.risk_level === "High" ? 0.95 : r.risk_level === "Medium" ? 0.75 : 0.45;
    }
    map.setFeatureState({ source: "roads", id: i }, { color, op });
  });
}

function applyHeatStates() {
  if (!S.mapReady) return;
  const tiles = S.data.cities[S.city].tiles;
  const c = cuts();
  tiles.forEach((t, i) => {
    const v = valueAt(t.series, Math.round(S.monthIdx));
    // normalize against the city's high cut so "1" means critical-band territory
    const norm = v == null ? 0 : Math.max(0, Math.min(1, v / Math.max(1e-6, c.high)));
    map.setFeatureState({ source: "heat", id: i }, { v: norm });
  });
}

/* ---------- hover ---------- */
function onRoadHover(e) {
  const f = e.features && e.features[0];
  if (!f) return;
  map.getCanvas().style.cursor = "pointer";
  if (S.hovered !== f.id) {
    if (S.hovered != null) map.setFeatureState({ source: "roads", id: S.hovered }, { hover: false });
    S.hovered = f.id;
    map.setFeatureState({ source: "roads", id: f.id }, { hover: true });
  }
  const r = S.roads[f.id];
  const col = TIER_COLOR[r.risk_level];
  const tierTxt = r.risk_level === "High" ? "CRITICAL" : r.risk_level === "Medium" ? "WATCH" : "STABLE";
  let timeRow = "";
  if (S.mode === "time") {
    const band = bandOf(valueAt(r.trend, Math.round(S.monthIdx)));
    timeRow = band ? `<div class="tt-row">${monthLabel(S.data.months[Math.round(S.monthIdx)])}: <b>${band === "High" ? "critical band" : band === "Medium" ? "watch band" : "stable"}</b></div>` : "";
  }
  tooltip.innerHTML = `
    <span class="tt-tier" style="background:${col}1f;color:${col};border:1px solid ${col}55">${tierTxt}</span>
    <div class="tt-name">${r.name}</div>
    <div class="tt-row">Priority score <b>${score100(r)} / 100</b> · ${(r.length_m / 1000).toFixed(1)} km</div>
    ${r.drivers && r.drivers[0] ? `<div class="tt-row">Main driver: <b>${r.drivers[0].label}</b></div>` : ""}
    ${timeRow}
    <div class="tt-row" style="color:${ACCENT}">Click for detail</div>`;
  tooltip.style.display = "block";
  const rect = map.getContainer().getBoundingClientRect();
  const px = e.point.x, py = e.point.y;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  tooltip.style.left = Math.min(rect.width - tw - 12, px + 14) + "px";
  tooltip.style.top = Math.max(8, py - th - 12) + "px";
}

function clearHover() {
  map.getCanvas().style.cursor = "";
  if (S.hovered != null) map.setFeatureState({ source: "roads", id: S.hovered }, { hover: false });
  S.hovered = null;
  tooltip.style.display = "none";
}

/* ============================================================
   DATA + BOOT
   ============================================================ */
async function loadData() {
  const bar = $("splash-progress");
  bar.style.width = "25%";
  let res;
  try {
    res = await fetch("/dashboard");
    if (!res.ok) throw new Error();
  } catch {
    res = await fetch("dashboard.json");
  }
  bar.style.width = "65%";
  S.data = await res.json();
  bar.style.width = "100%";

  const cityKeys = Object.keys(S.data.cities);
  S.city = cityKeys.includes("bengaluru") ? "bengaluru" : cityKeys[0];
  S.monthIdx = S.data.months.length - 1;

  buildCitySwitch(cityKeys);
  prepareCity();
  if (S.mapReady) loadCityIntoMap(false);
}

function prepareCity() {
  S.roads = S.data.roads.filter((r) => r.city === S.city);
  S.selected = null;
  S.hovered = null;
  $("drawer").classList.add("hidden");
  renderHeroStats();
  renderActFirst();
  renderScrubber();
  runBudget();
  updateTimeUI();
}

function buildCitySwitch(cityKeys) {
  const nav = $("city-switch");
  nav.innerHTML = "";
  for (const c of cityKeys) {
    const b = document.createElement("button");
    b.className = "seg-btn" + (c === S.city ? " active" : "");
    b.textContent = S.data.cities[c].label;
    b.onclick = () => {
      stopPlayback();
      S.city = c;
      nav.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      S.monthIdx = S.data.months.length - 1;
      prepareCity();
      loadCityIntoMap(true);
    };
    nav.appendChild(b);
  }
}

document.querySelectorAll("#mode-switch .seg-btn").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll("#mode-switch .seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    setMode(b.dataset.mode);
  };
});

function setMode(mode) {
  S.mode = mode;
  stopPlayback();
  $("panel-overview").classList.toggle("hidden", mode !== "overview");
  $("panel-story").classList.toggle("hidden", mode !== "time");
  $("panel-budget").classList.toggle("hidden", mode !== "budget");
  $("timebar").classList.toggle("hidden", mode !== "time");
  if (S.mapReady) map.setLayoutProperty("heat", "visibility", mode === "time" ? "visible" : "none");

  if (mode !== "time") S.monthIdx = S.data.months.length - 1;
  if (mode === "time") {
    S.monthIdx = 0; // the story starts at the beginning
    fitCity(true);
    renderScrubber();
  }
  if (mode === "budget") runBudget();
  applyRoadStates();
  applyHeatStates();
  updateTimeUI();
}

/* ============================================================
   OVERVIEW PANEL
   ============================================================ */
function renderHeroStats() {
  const sm = S.data.cities[S.city].summary;
  $("ov-kicker").textContent = `${S.data.cities[S.city].label.toUpperCase()} — RISK SNAPSHOT`;

  const exposure = sm.total_length_km * (sm.pct_high / 100) * COST_PER_KM_CR * (REACTIVE_MULT - 1) * FAIL_LIKELIHOOD;
  const highCount = S.roads.filter((r) => r.risk_level === "High").length;

  const firstCard = sm.critical_now > 0
    ? { color: "var(--critical)", num: sm.critical_now, cap: `road segments are in the <b>critical band today</b> and need attention this quarter` }
    : sm.entering_critical_6m > 0
      ? { color: "var(--critical)", num: sm.entering_critical_6m, cap: `road segments projected to <b>enter the critical band within 6 months</b>` }
      : { color: "var(--critical)", num: highCount, cap: `tracked road segments sit in the <b>top risk tier</b> — inspect these first` };

  const cards = [
    firstCard,
    {
      color: sm.monsoon_stress_yoy_pct > 0 ? "var(--critical)" : "var(--saved)",
      num: (sm.monsoon_stress_yoy_pct > 0 ? "+" : "") + (sm.monsoon_stress_yoy_pct ?? 0) + "%",
      raw: true,
      cap: `monsoon stress vs last year — ${sm.monsoon_stress_yoy_pct > 0 ? "rising, act before the next wet season" : "easing, hold preventive gains"}`,
    },
    {
      color: "var(--watch)",
      num: sm.risk_concentration ? sm.risk_concentration + "×" : "—",
      raw: true,
      cap: `the top 10% of roads carry <b>${sm.risk_concentration}× the stress</b> of the rest — small budgets go far here`,
    },
    {
      color: "var(--text)",
      num: fmtCr(Math.round(exposure)),
      raw: true,
      cap: `estimated reactive repair exposure if nothing is done · worst zone: <b>${zoneName(sm.worst_zone)}</b>`,
    },
  ];

  const wrap = $("hero-stats");
  wrap.innerHTML = "";
  for (const c of cards) {
    const el = document.createElement("div");
    el.className = "stat-card";
    el.innerHTML = `<div class="stat-num" style="color:${c.color}"></div><div class="stat-cap">${c.cap}</div>`;
    wrap.appendChild(el);
    const numEl = el.querySelector(".stat-num");
    if (c.raw) numEl.textContent = c.num;
    else countUp(numEl, c.num);
  }
}

function renderActFirst() {
  const list = $("act-first");
  list.innerHTML = "";
  S.roads
    .filter((r) => r.name !== "Unnamed Road")
    .sort((a, b) => b.risk_score - a.risk_score)
    .slice(0, 8)
    .forEach((r, i) => list.appendChild(roadItem(r, i + 1)));
}

function roadItem(r, rank, extra = "") {
  const li = document.createElement("li");
  li.className = "road-item";
  const col = TIER_COLOR[r.risk_level];
  li.innerHTML = `
    <span class="ri-rank">${rank ?? ""}</span>
    <span class="ri-body">
      <div class="ri-name">${r.name}</div>
      <div class="ri-sub">${zoneName(r.zone)} · ${(r.length_m / 1000).toFixed(1)} km ${extra}</div>
      <div class="ri-bar"><i style="width:${score100(r)}%;background:${col}"></i></div>
    </span>
    <span class="ri-risk" style="color:${col}">${score100(r)}</span>`;
  li.onclick = () => selectRoad(r);
  return li;
}

/* ============================================================
   ROAD SELECTION / DRAWER
   ============================================================ */
function selectRoad(r) {
  S.selected = r;
  map.getSource("selected").setData({
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: r.path } }],
  });

  let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
  for (const [lon, lat] of r.path) {
    minx = Math.min(minx, lon); maxx = Math.max(maxx, lon);
    miny = Math.min(miny, lat); maxy = Math.max(maxy, lat);
  }
  map.fitBounds([[minx, miny], [maxx, maxy]], {
    padding: { top: 80, bottom: 120, left: 380, right: 440 },
    maxZoom: 15,
    duration: 900,
  });

  const d = $("drawer");
  d.classList.remove("hidden");

  const tier = $("drawer-tier");
  tier.textContent = r.risk_level === "High" ? "TOP PRIORITY" : r.risk_level === "Medium" ? "WATCH LIST" : "STABLE";
  tier.className = "tier-badge tier-" + (r.risk_level === "High" ? "high" : r.risk_level === "Medium" ? "med" : "low");

  $("drawer-name").textContent = r.name;
  $("drawer-sub").textContent =
    `${S.data.cities[r.city].label} · ${zoneName(r.zone)} sector · city rank #${r.city_rank ?? r.rank}`;

  $("drawer-risk").textContent = score100(r);
  $("drawer-risk").style.color = TIER_COLOR[r.risk_level];

  const cd = $("drawer-countdown"), cdCap = $("drawer-countdown-cap");
  if (r.months_to_critical === 0) { cd.textContent = "NOW"; cd.style.color = "var(--critical)"; cdCap.textContent = "in critical band"; }
  else if (r.months_to_critical != null) { cd.textContent = r.months_to_critical + " mo"; cd.style.color = "var(--watch)"; cdCap.textContent = "to critical band"; }
  else { cd.textContent = "—"; cd.style.color = ""; cdCap.textContent = "no critical trajectory"; }

  $("drawer-len").textContent = (r.length_m / 1000).toFixed(1) + " km";

  const dw = $("drawer-drivers");
  dw.innerHTML = "";
  if (!r.drivers || !r.drivers.length) dw.innerHTML = `<div class="ri-sub">No dominant stress driver — risk is broad-based.</div>`;
  for (const drv of r.drivers || []) {
    const sev = drv.score >= 80 ? "severe" : drv.score >= 55 ? "elevated" : "moderate";
    const row = document.createElement("div");
    row.className = "driver-row";
    row.innerHTML = `
      <div class="driver-head"><b>${drv.label}</b><span class="${sev}">${sev}</span></div>
      <div class="driver-track"><div class="driver-fill ${sev}"></div></div>`;
    dw.appendChild(row);
    requestAnimationFrame(() => (row.querySelector(".driver-fill").style.width = drv.score + "%"));
  }

  drawSpark(r);
  $("drawer-action").textContent = r.action;
  const btn = $("add-plan-btn");
  const setBtn = () => (btn.textContent = S.plan.has(r.id) ? "✓ In maintenance plan" : "Add to maintenance plan");
  setBtn();
  btn.onclick = () => {
    if (S.plan.has(r.id)) { S.plan.delete(r.id); toast(`${r.name} removed from plan`); }
    else { S.plan.add(r.id); toast(`${r.name} added to the maintenance plan`); }
    setBtn();
    applyRoadStates();
  };

  updateStoryRoad();
}

$("drawer-close").onclick = () => {
  $("drawer").classList.add("hidden");
  S.selected = null;
  map.getSource("selected").setData(emptyFC());
  updateStoryRoad();
};

/* Sparkline: smoothed trend, monsoon bands, critical threshold. */
function drawSpark(r) {
  const c = $("spark");
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = c.clientWidth || 360, h = 90;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const smoothed = r.trend.map((v, i) => {
    if (v == null) return null;
    const win = [];
    for (let j = Math.max(0, i - 2); j <= i; j++) if (r.trend[j] != null) win.push(r.trend[j]);
    return win.reduce((a, b) => a + b, 0) / win.length;
  });
  const vals = smoothed.map((v, i) => [i, v]).filter(([, v]) => v != null);
  if (vals.length < 2) { g.fillStyle = "#8b929b"; g.font = "11px sans-serif"; g.fillText("Not enough history", 10, 45); return; }

  const n = r.trend.length;
  const highCut = cuts().high;
  const lo = Math.min(...vals.map((v) => v[1]), highCut);
  const hi = Math.max(...vals.map((v) => v[1]), highCut);
  const X = (i) => 6 + (i / (n - 1)) * (w - 12);
  const Y = (v) => h - 16 - ((v - lo) / Math.max(1e-6, hi - lo)) * (h - 34);

  // monsoon bands
  g.fillStyle = "rgba(217,154,43,0.07)";
  S.data.months.forEach((m, i) => {
    if (isMonsoon(m)) g.fillRect(X(i) - (w / n) / 2, 6, w / n, h - 22);
  });

  // critical threshold
  g.strokeStyle = "rgba(229,72,77,0.55)";
  g.setLineDash([4, 4]);
  g.beginPath(); g.moveTo(6, Y(highCut)); g.lineTo(w - 6, Y(highCut)); g.stroke();
  g.setLineDash([]);
  g.fillStyle = "rgba(229,72,77,0.8)";
  g.font = "9px sans-serif";
  g.fillText("critical band", 8, Y(highCut) - 4);

  // trend line — single neutral color; the threshold gives it meaning
  g.strokeStyle = "#b9bfc7";
  g.lineWidth = 1.8;
  g.lineCap = "round";
  g.beginPath();
  let prev = null;
  for (const [i, v] of vals) {
    if (prev && i - prev[0] > 3) prev = null;
    if (prev) g.lineTo(X(i), Y(v));
    else g.moveTo(X(i), Y(v));
    prev = [i, v];
  }
  g.stroke();

  const last = vals[vals.length - 1];
  const lastBand = bandOf(last[1]);
  g.fillStyle = TIER_COLOR[lastBand ?? "Low"];
  g.beginPath(); g.arc(X(last[0]), Y(last[1]), 3.2, 0, 7); g.fill();

  g.fillStyle = "#8b929b"; g.font = "9.5px sans-serif";
  g.fillText(monthLabel(S.data.months[0]), 6, h - 3);
  const lm = monthLabel(S.data.months[n - 1]);
  g.fillText(lm, w - g.measureText(lm).width - 6, h - 3);
}

/* ============================================================
   SEARCH
   ============================================================ */
$("search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  const box = $("search-results");
  if (q.length < 2) { box.classList.remove("open"); return; }
  const hits = S.roads
    .filter((r) => r.name.toLowerCase().includes(q) && r.name !== "Unnamed Road")
    .sort((a, b) => b.risk_score - a.risk_score)
    .slice(0, 12);
  box.innerHTML = hits.length ? "" : `<div class="sr-item">No roads found</div>`;
  for (const r of hits) {
    const div = document.createElement("div");
    div.className = "sr-item";
    div.innerHTML = `<span>${r.name}</span><span class="sr-risk" style="color:${TIER_COLOR[r.risk_level]}">${score100(r)}</span>`;
    div.onclick = () => { box.classList.remove("open"); $("search").value = ""; selectRoad(r); };
    box.appendChild(div);
  }
  box.classList.add("open");
});
document.addEventListener("click", (e) => {
  if (!$("search-wrap").contains(e.target)) $("search-results").classList.remove("open");
});

/* ============================================================
   TIME MACHINE — playback + narrative
   ============================================================ */
$("play-btn").onclick = () => {
  if (S.playTimer) { stopPlayback(); return; }
  if (S.monthIdx >= S.data.months.length - 1) S.monthIdx = 0;
  $("play-btn").textContent = "❚❚";
  S.playTimer = setInterval(() => {
    if (S.monthIdx >= S.data.months.length - 1) { stopPlayback(); return; }
    S.monthIdx += 1;
    updateTimeUI();
    applyRoadStates();
    applyHeatStates();
  }, PLAY_MS_PER_MONTH);
};

function stopPlayback() {
  clearInterval(S.playTimer);
  S.playTimer = null;
  $("play-btn").textContent = "▶";
}

$("scrubber").addEventListener("input", (e) => {
  stopPlayback();
  S.monthIdx = +e.target.value;
  updateTimeUI();
  applyRoadStates();
  applyHeatStates();
});

function hotCountAt(i) {
  const c = cuts();
  let n = 0;
  for (const r of S.roads) {
    const v = valueAt(r.trend, i);
    if (v != null && v >= c.high) n++;
  }
  return n;
}

function updateTimeUI() {
  if (!S.data) return;
  const i = Math.round(S.monthIdx);
  const m = S.data.months[i];
  $("time-month").textContent = monthLabel(m);
  const note = $("time-note");
  note.textContent = isMonsoon(m) ? "monsoon window — stress peaks" : "dry season";
  note.classList.toggle("monsoon", isMonsoon(m));
  $("scrubber").value = i;
  drawScrubCursor();

  if (S.mode !== "time") return;

  const ss = S.data.cities[S.city].stress_series || {};
  const rain = valueAt(ss.rain || [], i);
  const flood = valueAt(ss.flood || [], i);
  const heat = valueAt(ss.heat || [], i);

  setMeter("rain", rain);
  setMeter("flood", flood);
  setMeter("heat", heat);

  $("story-month").textContent = monthLabel(m);
  const phase = $("story-phase");
  const mn = monthNum(m);
  if (isMonsoon(m)) { phase.textContent = "MONSOON WINDOW"; phase.className = "monsoon"; }
  else if (mn >= 3 && mn <= 5) { phase.textContent = "PRE-MONSOON HEAT"; phase.className = ""; }
  else { phase.textContent = "DRY SEASON"; phase.className = ""; }

  const hot = hotCountAt(i);
  const hotPrev = i >= 6 ? hotCountAt(i - 6) : null;
  $("hot-count").textContent = hot;
  $("hot-delta").innerHTML = hotPrev != null
    ? (hot > hotPrev
      ? `up from <b>${hotPrev}</b> six months ago — the network is losing ground`
      : hot < hotPrev
        ? `down from <b>${hotPrev}</b> six months ago — pressure is easing`
        : `unchanged over six months`)
    : "";

  $("story-caption").textContent = storyCaption(i, m, { rain, flood, heat }, hot, hotPrev);
  updateStoryRoad();
}

function setMeter(key, v) {
  const val = v == null ? 0 : v;
  $("mv-" + key).textContent = v == null ? "—" : v + " / 100";
  const bar = $("mf-" + key);
  bar.style.width = val + "%";
  bar.className = val >= 75 ? "severe" : val >= 50 ? "hot" : "";
}

function worstZoneAt(i) {
  const tiles = S.data.cities[S.city].tiles;
  let best = null, bestV = -1;
  for (const t of tiles) {
    const v = valueAt(t.series, i);
    if (v != null && v > bestV) { bestV = v; best = t; }
  }
  return best ? zoneName(zoneOfTile(best.id)) : null;
}

function storyCaption(i, m, d, hot, hotPrev) {
  const zone = worstZoneAt(i);
  const zoneTxt = zone ? ` The ${zone} sector is taking the worst of it.` : "";
  const worsening = hotPrev != null && hot > hotPrev;

  if (isMonsoon(m)) {
    if ((d.rain ?? 0) >= 50) {
      return `Monsoon rain is loading the network — rainfall at ${d.rain}/100` +
        ((d.flood ?? 0) >= 50 ? ` with standing water at ${d.flood}/100. Water works into cracks and weakens the road base.` : `. Saturated ground accelerates surface wear.`) +
        zoneTxt;
    }
    return `The monsoon window is open, though rainfall loading is moderate so far (${d.rain ?? "—"}/100). Each wet spell soaks into cracks left by earlier seasons — drainage fixes made now pay off double.` + zoneTxt;
  }
  if ((d.flood ?? 0) >= 65) {
    return `Standing water is the dominant pressure right now (${d.flood}/100). Poor drainage keeps moisture in the pavement structure long after the rain stops.` + zoneTxt;
  }
  if ((d.heat ?? 0) >= 60 && !isMonsoon(m)) {
    return `Dry-season heat is doing the damage now — thermal stress at ${d.heat}/100 expands and cracks surfacing that the last monsoon weakened.` + zoneTxt;
  }
  if (worsening) {
    return `No single driver dominates, but damage is compounding: each wet-dry cycle leaves roads slightly weaker than the one before.` + zoneTxt;
  }
  return `Stress is comparatively low this month — the best window for inspections and preventive resurfacing before the next monsoon.`;
}

function updateStoryRoad() {
  const block = $("story-road");
  if (S.mode !== "time" || !S.selected) { block.classList.add("hidden"); return; }
  block.classList.remove("hidden");
  const r = S.selected;
  const i = Math.round(S.monthIdx);
  const v = valueAt(r.trend, i);
  const band = bandOf(v);
  $("story-road-name").textContent = r.name;
  const driver = r.drivers && r.drivers[0] ? r.drivers[0].label.toLowerCase() : null;
  let status;
  if (band == null) status = "No satellite reading for this month.";
  else if (band === "High") status = `In the <b>critical band</b> this month${driver ? ` — ${driver} has taken its toll` : ""}.`;
  else if (band === "Medium") status = `In the <b>watch band</b>${driver ? ` — ${driver} is building up` : ""}.`;
  else status = `Holding <b>stable</b> at this point in time.`;
  $("story-road-status").innerHTML = status;
}

/* Scrubber backdrop: city risk line + monsoon bands + cursor. */
let scrubBase = null; // cached ImageBitmap-less redraw fn
function cityRiskSeries() {
  const tiles = S.data.cities[S.city].tiles;
  const n = S.data.months.length;
  const out = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (const t of tiles) { const v = t.series[i]; if (v != null) { s += v; c++; } }
    out[i] = c ? s / c : null;
  }
  return out;
}

function renderScrubber() {
  const c = $("trend-canvas");
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = c.clientWidth || 600, h = 56;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  c._w = w; c._h = h;
  drawScrubBase(g, w, h);
  drawScrubCursor();
}

function drawScrubBase(g, w, h) {
  g.clearRect(0, 0, w, h);
  const series = cityRiskSeries();
  const vals = series.filter((v) => v != null);
  if (!vals.length) return;
  const min = Math.min(...vals), max = Math.max(...vals);
  const X = (i) => (i / (series.length - 1)) * w;
  const Y = (v) => h - 6 - ((v - min) / Math.max(1e-6, max - min)) * (h - 16);

  g.fillStyle = "rgba(217,154,43,0.08)";
  S.data.months.forEach((m, i) => {
    if (isMonsoon(m)) g.fillRect(X(i) - w / series.length / 2, 4, w / series.length, h - 10);
  });

  g.strokeStyle = "#8b929b";
  g.lineWidth = 1.4;
  g.beginPath();
  let started = false;
  series.forEach((v, i) => {
    if (v == null) return;
    if (!started) { g.moveTo(X(i), Y(v)); started = true; }
    else g.lineTo(X(i), Y(v));
  });
  g.stroke();
  // stash for cursor redraws
  scrubBase = g.getImageData(0, 0, g.canvas.width, g.canvas.height);
}

function drawScrubCursor() {
  const c = $("trend-canvas");
  if (!c._w || !scrubBase) return;
  const g = c.getContext("2d");
  g.putImageData(scrubBase, 0, 0);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const i = Math.round(S.monthIdx);
  const x = (i / (S.data.months.length - 1)) * c._w;
  g.save();
  g.scale(dpr, dpr);
  g.strokeStyle = ACCENT;
  g.lineWidth = 1.4;
  g.beginPath(); g.moveTo(x, 2); g.lineTo(x, c._h - 2); g.stroke();
  g.restore();
}

/* ============================================================
   BUDGET SIMULATOR
   ============================================================ */
function budgetPool() {
  return S.roads
    .filter((r) => r.risk_level !== "Low")
    .sort((a, b) =>
      (b.risk_level === "High") - (a.risk_level === "High") ||
      b.risk_score - a.risk_score ||
      (b.name !== "Unnamed Road") - (a.name !== "Unnamed Road"));
}

function runBudget() {
  const slider = $("budget-slider");
  const pool = budgetPool();
  const totalCost = pool.reduce((a, r) => a + (r.length_m / 1000) * COST_PER_KM_CR, 0);
  slider.max = Math.max(10, Math.ceil(totalCost / 5) * 5);
  $("budget-max-label").textContent = fmtCr(+slider.max);
  const budget = Math.min(S.budgetCr, +slider.max);
  slider.value = budget;
  slider.style.setProperty("--fill", (budget / slider.max) * 100 + "%");
  $("budget-value").textContent = fmtCr(budget);

  let spent = 0;
  const picks = [];
  for (const r of pool) {
    const cost = (r.length_m / 1000) * COST_PER_KM_CR;
    if (spent + cost > budget) continue;
    spent += cost;
    picks.push({ road: r, cost });
  }
  S.budgetPicks = picks.map((p) => p.road);
  if (S.mode === "budget") applyRoadStates();

  const km = picks.reduce((a, p) => a + p.road.length_m / 1000, 0);
  const highs = S.roads.filter((r) => r.risk_level === "High");
  const highPicked = picks.filter((p) => p.road.risk_level === "High").length;
  const cover = highs.length ? (highPicked / highs.length) * 100 : 0;
  const saved = km * COST_PER_KM_CR * (REACTIVE_MULT - 1) * FAIL_LIKELIHOOD;

  countUp($("bk-roads"), picks.length);
  countUp($("bk-km"), km, { decimals: 1 });
  countUp($("bk-saved"), saved, { prefix: "₹", suffix: " cr", decimals: saved < 10 ? 1 : 0 });
  countUp($("bk-cover"), cover, { suffix: "%", decimals: 0 });
  $("coverage-fill").style.width = cover + "%";
  $("roi-line").innerHTML = spent > 0
    ? `Every ₹1 spent now avoids an estimated <b>₹${(saved / Math.max(0.01, spent)).toFixed(1)}</b> in reactive repairs.`
    : `Move the slider to see what early action buys.`;

  const list = $("plan-list");
  list.innerHTML = "";
  picks.slice(0, 30).forEach((p, i) => {
    list.appendChild(roadItem(p.road, i + 1, `· <span class="plan-cost">${fmtCr(p.cost)}</span>`));
  });
  if (picks.length > 30) {
    const more = document.createElement("li");
    more.className = "ri-sub";
    more.style.padding = "6px 10px";
    more.textContent = `+ ${picks.length - 30} more roads in the full work order (see Executive Brief)`;
    list.appendChild(more);
  }
}

$("budget-slider").addEventListener("input", (e) => {
  S.budgetCr = +e.target.value;
  runBudget();
});

/* ============================================================
   EXECUTIVE BRIEF
   ============================================================ */
$("brief-btn").onclick = buildBrief;
$("brief-close").onclick = () => $("brief-overlay").classList.add("hidden");
$("brief-print").onclick = () => window.print();

function buildBrief() {
  const sm = S.data.cities[S.city].summary;
  const label = S.data.cities[S.city].label;
  const top = S.roads.filter((r) => r.name !== "Unnamed Road").sort((a, b) => b.risk_score - a.risk_score).slice(0, 20);
  const planRoads = S.budgetPicks.length ? S.budgetPicks : top.filter((r) => r.risk_level === "High");
  const planKm = planRoads.reduce((a, r) => a + r.length_m / 1000, 0);
  const planCost = planKm * COST_PER_KM_CR;
  const planSaved = planKm * COST_PER_KM_CR * (REACTIVE_MULT - 1) * FAIL_LIKELIHOOD;
  const now = new Date();

  const rows = top.map((r, i) => `
    <tr>
      <td>${i + 1}</td><td><b>${r.name}</b></td><td>${zoneName(r.zone)}</td>
      <td>${(r.length_m / 1000).toFixed(1)} km</td>
      <td><span class="chip chip-${r.risk_level === "High" ? "high" : r.risk_level === "Medium" ? "med" : "low"}">${score100(r)} / 100</span></td>
      <td>${r.action}</td>
    </tr>`).join("");

  $("brief-content").innerHTML = `
    <div class="brief-header">
      <h1>Road Infrastructure Risk Brief — ${label}</h1>
      <div class="bh-sub">Prepared ${now.toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" })} ·
      Satellite-observed stress, 2020–2024 · For maintenance planning purposes</div>
    </div>

    <div class="brief-kpis">
      <div class="brief-kpi"><div class="n">${sm.critical_now > 0 ? sm.critical_now : S.roads.filter((r) => r.risk_level === "High").length}</div>
        <div class="c">${sm.critical_now > 0 ? "segments in critical band now" : "segments in the top priority tier"}</div></div>
      <div class="brief-kpi"><div class="n">${sm.monsoon_stress_yoy_pct > 0 ? "+" : ""}${sm.monsoon_stress_yoy_pct ?? 0}%</div><div class="c">monsoon stress vs last year</div></div>
      <div class="brief-kpi"><div class="n">${sm.risk_concentration ?? "—"}×</div><div class="c">stress concentration in top 10% of roads</div></div>
      <div class="brief-kpi"><div class="n">${zoneName(sm.worst_zone)}</div><div class="c">most stressed sector</div></div>
    </div>

    <div class="brief-section">
      <h2>RECOMMENDED INVESTMENT</h2>
      <p style="font-size:13px;line-height:1.7">
        Treating the <b>${planRoads.length} highest-priority roads</b> (${planKm.toFixed(1)} km) with preventive
        maintenance requires an estimated <b>${fmtCr(planCost)}</b> and avoids an estimated
        <b>${fmtCr(planSaved)}</b> in reactive repairs over the following three years —
        <b>₹${(planSaved / Math.max(0.01, planCost)).toFixed(1)} avoided per ₹1 invested</b>.
      </p>
    </div>

    <div class="brief-section">
      <h2>TOP 20 PRIORITY ROADS</h2>
      <table class="brief-table">
        <thead><tr><th>#</th><th>ROAD</th><th>SECTOR</th><th>LENGTH</th><th>PRIORITY</th><th>RECOMMENDED ACTION</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>

    <div class="brief-section">
      <h2>SIX-MONTH ACTION PLAN</h2>
      <table class="brief-table">
        <thead><tr><th>WINDOW</th><th>FOCUS</th></tr></thead>
        <tbody>
          <tr><td><b>0–30 days</b></td><td>Inspect all critical-band segments; drainage audits where water accumulation is the lead driver.</td></tr>
          <tr><td><b>30–90 days</b></td><td>Tender preventive resurfacing for the work order above; prioritise the ${zoneName(sm.worst_zone)} sector.</td></tr>
          <tr><td><b>90–180 days</b></td><td>Complete resurfacing before monsoon onset; re-assess watch-list roads with fresh satellite passes.</td></tr>
        </tbody>
      </table>
    </div>

    <div class="brief-note">
      How to read this brief: risk scores rank roads <i>relative to each other</i> within ${label} using
      satellite-observed environmental stress (rainfall loading, standing water, heat exposure, surface moisture,
      corridor usage). Scores prioritise inspection and preventive spending; they are not structural assessments.
      Cost figures use planning assumptions (preventive resurfacing ₹${(COST_PER_KM_CR * 100).toFixed(0)} lakh/km,
      reactive repairs ${REACTIVE_MULT}× preventive) and should be refined with department rate cards.
    </div>`;

  $("brief-overlay").classList.remove("hidden");
}

/* ---------- boot ---------- */
// In throttled/background contexts rAF can stall entirely, which freezes
// MapLibre's style pipeline. Fall back to a timer-driven frame source.
function ensureRAF() {
  return new Promise((resolve) => {
    let fired = false;
    requestAnimationFrame(() => { fired = true; resolve(); });
    setTimeout(() => {
      if (fired) return;
      window.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 16);
      window.cancelAnimationFrame = (id) => clearTimeout(id);
      resolve();
    }, 250);
  });
}

ensureRAF().then(() => initMap());
loadData().catch((err) => {
  $("splash-progress").style.width = "100%";
  document.querySelector(".splash-sub").textContent = "COULD NOT LOAD RISK DATA — IS THE API RUNNING?";
  console.error(err);
});
window.addEventListener("resize", () => { if (S.data) renderScrubber(); });
