/**
 * LoL Player Props — Google Sheets Script
 * ========================================
 * HOW TO INSTALL:
 *   1. Extensions → Apps Script → delete everything → paste this
 *   2. Change API_URL on line 11 to your Railway URL
 *   3. Ctrl+S → close tab → reload sheet
 *   4. LoL Props → Setup Sheet
 */

const API_URL = "https://lol-production-a106.up.railway.app"; // ← your Railway URL

// ─── Column layout ─────────────────────────────────────────────────────────────
// INPUTS        A=Player  B=Side  C=Team ML  D=Opp ML  E=Format
// SPORTSBOOK    F=SB Kills  G=SB Deaths  H=SB Assists  (lines you type in)
// MODEL M1      I=Kills  J=Deaths  K=Assists
// MODEL M1-2    L=Kills  M=Deaths  N=Assists  O=Fantasy
// MODEL M1-3    P=Kills  Q=Deaths  R=Assists  S=Fantasy
// EDGE          T=Kills Edge  U=Deaths Edge  V=Assists Edge
// META          W=Position  X=League  Y=Win%  Z=Recent Form  AA=Updated

const C = {
  // Inputs
  PLAYER:1, SIDE:2, TEAM_ML:3, OPP_ML:4, FORMAT:5, OPP_TEAM:6,
  // Sportsbook lines (you fill these)
  SB_KILLS:7, SB_DEATHS:8, SB_ASSISTS:9,
  // Model predictions
  M1_KILLS:10,  M1_DEATHS:11, M1_ASSISTS:12,
  M12_KILLS:13, M12_DEATHS:14, M12_ASSISTS:15, M12_FANTASY:16,
  M13_KILLS:17, M13_DEATHS:18, M13_ASSISTS:19, M13_FANTASY:20,
  // Edge (model - sportsbook line)
  EDGE_KILLS:21, EDGE_DEATHS:22, EDGE_ASSISTS:23,
  // Meta
  POSITION:24, LEAGUE:25, WIN_PCT:26, GAMES:27, RECENT:28, UPDATED:29
};

const DATA_START = 3;

// ─── Menu ──────────────────────────────────────────────────────────────────────
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("LoL Props")
    .addItem("🎯 Show best bets (strong unders)","showBestBets")
    .addItem("🔍 Search & predict player",      "searchAndPredict")
    .addItem("▶  Predict selected rows",        "predictSelected")
    .addItem("▶  Predict ALL rows",             "predictAll")
    .addItem("📊 Recalculate edges",            "recalcEdges")
    .addSeparator()
    .addItem("🔄 Refresh data from OE sheet",   "refreshData")
    .addSeparator()
    .addItem("📊 Log selected predictions",     "logSelectedPredictions")
    .addItem("✅ Update hit/miss results",      "updateHitMiss")
    .addItem("📈 Accuracy summary",             "getModelAccuracySummary")
    .addItem("🗂️  Setup tracking sheet",        "setupTrackingSheet")
    .addSeparator()
    .addItem("📋 Setup sheet",                  "setupSheet")
    .addItem("🩺 Check API status",             "checkStatus")
    .addToUi();
}

// ─── API ───────────────────────────────────────────────────────────────────────
function apiGet(path) {
  const resp = UrlFetchApp.fetch(API_URL + path, {
    method: "GET", muteHttpExceptions: true,
    headers: { Accept: "application/json" }
  });
  const code = resp.getResponseCode();
  const text = resp.getContentText();
  if (code !== 200) {
    let msg = `HTTP ${code}: ${text.substring(0, 300)}`;
    try { msg = JSON.parse(text).detail?.message || JSON.parse(text).detail || msg; } catch(e) {}
    throw new Error(msg);
  }
  try { return JSON.parse(text); } catch(e) { throw new Error("Bad JSON: " + text.substring(0,200)); }
}

function fetchPrediction(player, side, teamML, oppML, oppTeam) {
  let path = `/predict?player=${encodeURIComponent(player)}&side=${encodeURIComponent(side||"Blue")}`;
  const ml = parseInt(teamML), oml = parseInt(oppML);
  if (!isNaN(ml) && !isNaN(oml)) path += `&moneyline=${ml}&opp_ml=${oml}`;
  if (oppTeam && oppTeam.toString().trim() !== "") path += `&opponent=${encodeURIComponent(oppTeam.toString().trim())}`;
  return apiGet(path);
}

// ─── Edge calculation ──────────────────────────────────────────────────────────
// Edge = model projection - sportsbook line
// Positive = model says OVER, Negative = model says UNDER
// Colour: strong green = big over edge, strong red = big under edge

function calcEdge(modelVal, sbLine) {
  if (sbLine === "" || sbLine === null || isNaN(parseFloat(sbLine))) return null;
  return round2(modelVal - parseFloat(sbLine));
}

function round2(v) { return Math.round(v * 100) / 100; }

function colourEdge(range, edge) {
  if (edge === null) { range.setBackground("#ffffff"); range.setFontWeight("normal"); range.setFontColor("#000000"); range.setFontLine("none"); return; }
  const abs = Math.abs(edge);

  if (abs < 2.0) {
    // Weak edge — grey, don't bet
    range.setBackground("#f5f5f5");
    range.setFontColor("#9e9e9e");
    range.setFontWeight("normal");
    range.setFontLine("none");
  } else if (edge < 0) {
    // Strong UNDER edge (2+) — green, reliable
    const intensity = Math.min(1, (abs - 2.0) / 5.0);
    const r = Math.round(200 - 100 * intensity);
    const g = Math.round(220 + 35 * intensity);
    range.setBackground(`rgb(${r},${g},${r})`);
    range.setFontColor("#1b5e20");
    range.setFontWeight(abs >= 5.0 ? "bold" : "normal");
    range.setFontLine("none");
  } else {
    // Strong OVER edge (2+) — blue, also reliable based on data
    const intensity = Math.min(1, (abs - 2.0) / 5.0);
    const b = Math.round(220 + 35 * intensity);
    const rg = Math.round(200 - 100 * intensity);
    range.setBackground(`rgb(${rg},${rg},${b})`);
    range.setFontColor("#0d47a1");
    range.setFontWeight(abs >= 5.0 ? "bold" : "normal");
    range.setFontLine("none");
  }
}

function colourValue(range, val, min, max) {
  const pct = Math.min(1, Math.max(0, (val-min)/(max-min)));
  range.setBackground(`rgb(${Math.round(255*(1-pct))},${Math.round(200*pct)},60)`);
}

// ─── Get model value for current format ───────────────────────────────────────
function getModelVal(d, stat, format) {
  // For edge comparison, use the projection matching the format
  const fmt = (format || "M1-3").toString().toUpperCase().replace(/\s/g,"");
  if (fmt === "M1" || fmt === "MAP1") {
    return d.map1?.[stat]?.expected ?? null;
  } else if (fmt === "M1-2" || fmt === "M12") {
    return d.m1_2?.[stat]?.series_total ?? null;
  } else {
    return d.m1_3?.[stat]?.series_total ?? null;  // default M1-3
  }
}

// ─── Write row ─────────────────────────────────────────────────────────────────
function writeRow(sheet, row, d) {
  const m1  = d.map1 || {};
  const m12 = d.m1_2 || {};
  const m13 = d.m1_3 || {};

  const format  = sheet.getRange(row, C.FORMAT).getValue() || "M1-3";
  const recentStr = (d.recent_form||[]).slice(0,3)
    .map(g=>`${g.champion} ${g.kills}/${g.deaths}/${g.assists}`).join(", ");
  const winPct = d.win_prob != null ? (d.win_prob*100).toFixed(0)+"%" : "–";

  // Model predictions
  sheet.getRange(row, C.M1_KILLS).setValue(m1.kills?.expected ?? "–");
  sheet.getRange(row, C.M1_DEATHS).setValue(m1.deaths?.expected ?? "–");
  sheet.getRange(row, C.M1_ASSISTS).setValue(m1.assists?.expected ?? "–");

  sheet.getRange(row, C.M12_KILLS).setValue(m12.kills?.series_total ?? "–");
  sheet.getRange(row, C.M12_DEATHS).setValue(m12.deaths?.series_total ?? "–");
  sheet.getRange(row, C.M12_ASSISTS).setValue(m12.assists?.series_total ?? "–");
  sheet.getRange(row, C.M12_FANTASY).setValue(m12.fantasy ?? "–");

  sheet.getRange(row, C.M13_KILLS).setValue(m13.kills?.series_total ?? "–");
  sheet.getRange(row, C.M13_DEATHS).setValue(m13.deaths?.series_total ?? "–");
  sheet.getRange(row, C.M13_ASSISTS).setValue(m13.assists?.series_total ?? "–");
  sheet.getRange(row, C.M13_FANTASY).setValue(m13.fantasy ?? "–");

  // Meta
  sheet.getRange(row, C.POSITION).setValue(d.position || "–");
  sheet.getRange(row, C.LEAGUE).setValue(d.league || "–");
  sheet.getRange(row, C.WIN_PCT).setValue(winPct);
  const games = d.games_in_sample || 0;
  sheet.getRange(row, C.GAMES).setValue(games);
  // Colour code by sample size
  const gamesCell = sheet.getRange(row, C.GAMES);
  if (games > 0 && games < 10) gamesCell.setBackground("#ffcdd2").setFontWeight("bold");
  else if (games < 20) gamesCell.setBackground("#fff9c4").setFontWeight("normal");
  else gamesCell.setBackground("#c8e6c9").setFontWeight("normal");

  sheet.getRange(row, C.RECENT).setValue(recentStr);
  sheet.getRange(row, C.UPDATED).setValue(new Date().toLocaleString());

  // Colour model values
  try {
    if (m13.kills) {
      colourValue(sheet.getRange(row, C.M13_KILLS),   m13.kills.series_total,   0, 20);
      colourValue(sheet.getRange(row, C.M13_ASSISTS), m13.assists.series_total, 0, 35);
      colourValue(sheet.getRange(row, C.M12_KILLS),   m12.kills?.series_total||0, 0, 14);
      colourValue(sheet.getRange(row, C.M12_ASSISTS), m12.assists?.series_total||0, 0, 25);
    }
  } catch(e) {}

  // Calculate and write edges
  updateEdges(sheet, row, d);
}

// ─── Edge update (can be called standalone) ───────────────────────────────────
function updateEdges(sheet, row, d) {
  const format = sheet.getRange(row, C.FORMAT).getValue() || "M1-3";

  const sbKills   = parseFloat(sheet.getRange(row, C.SB_KILLS).getValue());
  const sbDeaths  = parseFloat(sheet.getRange(row, C.SB_DEATHS).getValue());
  const sbAssists = parseFloat(sheet.getRange(row, C.SB_ASSISTS).getValue());

  // Get model values — from live data object or from sheet cells
  let modelKills, modelDeaths, modelAssists;
  if (d) {
    modelKills   = getModelVal(d, "kills",   format);
    modelDeaths  = getModelVal(d, "deaths",  format);
    modelAssists = getModelVal(d, "assists", format);
  } else {
    // Read from correct column based on format
    const fmt = (format||"M1-3").toString().toUpperCase().replace(/[\s-]/g,"");
    if (fmt === "M1" || fmt === "MAP1") {
      modelKills   = parseFloat(sheet.getRange(row, C.M1_KILLS).getValue());
      modelDeaths  = parseFloat(sheet.getRange(row, C.M1_DEATHS).getValue());
      modelAssists = parseFloat(sheet.getRange(row, C.M1_ASSISTS).getValue());
    } else if (fmt === "M12") {
      modelKills   = parseFloat(sheet.getRange(row, C.M12_KILLS).getValue());
      modelDeaths  = parseFloat(sheet.getRange(row, C.M12_DEATHS).getValue());
      modelAssists = parseFloat(sheet.getRange(row, C.M12_ASSISTS).getValue());
    } else {
      modelKills   = parseFloat(sheet.getRange(row, C.M13_KILLS).getValue());
      modelDeaths  = parseFloat(sheet.getRange(row, C.M13_DEATHS).getValue());
      modelAssists = parseFloat(sheet.getRange(row, C.M13_ASSISTS).getValue());
    }
  }

  const edgeK = (!isNaN(modelKills)   && !isNaN(sbKills))   ? round2(modelKills   - sbKills)   : null;
  const edgeD = (!isNaN(modelDeaths)  && !isNaN(sbDeaths))  ? round2(modelDeaths  - sbDeaths)  : null;
  const edgeA = (!isNaN(modelAssists) && !isNaN(sbAssists)) ? round2(modelAssists - sbAssists) : null;

  const cellK = sheet.getRange(row, C.EDGE_KILLS);
  const cellD = sheet.getRange(row, C.EDGE_DEATHS);
  const cellA = sheet.getRange(row, C.EDGE_ASSISTS);

  // Force number format to prevent Google Sheets date auto-formatting
  const numFmt = "0.00";
  cellK.setNumberFormat(numFmt);
  cellD.setNumberFormat(numFmt);
  cellA.setNumberFormat(numFmt);

  cellK.setValue(edgeK !== null ? edgeK : "–");
  cellD.setValue(edgeD !== null ? edgeD : "–");
  cellA.setValue(edgeA !== null ? edgeA : "–");

  colourEdge(cellK, edgeK);
  colourEdge(cellD, edgeD);
  colourEdge(cellA, edgeA);
}

// ─── Recalculate all edges (when you update SB lines) ─────────────────────────
function recalcEdges() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const last  = sheet.getLastRow();
  let updated = 0;
  for (let r = DATA_START; r <= last; r++) {
    const player = sheet.getRange(r, C.PLAYER).getValue();
    if (!player) continue;
    const modelKills = sheet.getRange(r, C.M13_KILLS).getValue();
    if (!modelKills || modelKills === "–") continue;
    updateEdges(sheet, r, null);
    updated++;
  }
  SpreadsheetApp.getActive().toast(`✅ Edges recalculated for ${updated} players`, "LoL Props", 3);
}

// ─── Inputs from row ──────────────────────────────────────────────────────────
function rowInputs(sheet, row) {
  return {
    player:  sheet.getRange(row, C.PLAYER).getValue().toString().trim(),
    side:    sheet.getRange(row, C.SIDE).getValue().toString().trim() || "Blue",
    teamML:  sheet.getRange(row, C.TEAM_ML).getValue(),
    oppML:   sheet.getRange(row, C.OPP_ML).getValue(),
    format:  sheet.getRange(row, C.FORMAT).getValue().toString().trim() || "M1-3",
    oppTeam: sheet.getRange(row, C.OPP_TEAM).getValue().toString().trim(),
  };
}

// ─── Search & predict ─────────────────────────────────────────────────────────
function searchAndPredict() {
  const ui    = SpreadsheetApp.getUi();
  const sheet = SpreadsheetApp.getActiveSheet();

  const nameResp = ui.prompt("Search Player", "Type part of a name (e.g. 'fak'):", ui.ButtonSet.OK_CANCEL);
  if (nameResp.getSelectedButton() !== ui.Button.OK) return;
  const query = nameResp.getResponseText().trim();
  if (!query) return;

  let results;
  try { results = apiGet(`/search?q=${encodeURIComponent(query)}`); }
  catch(e) { ui.alert("Search failed: " + e.message); return; }
  if (!results?.length) { ui.alert("No players found: " + query); return; }

  const listText = results.map((p,i) =>
    `${i+1}. ${p.playername} — ${p.position} — ${p.league} (${p.games} games)`
  ).join("\n");

  const pickResp = ui.prompt("Pick a Player", `${listText}\n\nEnter number:`, ui.ButtonSet.OK_CANCEL);
  if (pickResp.getSelectedButton() !== ui.Button.OK) return;
  const pick = parseInt(pickResp.getResponseText().trim()) - 1;
  if (isNaN(pick) || pick < 0 || pick >= results.length) { ui.alert("Invalid."); return; }
  const chosen = results[pick];

  const mlResp = ui.prompt("Team Moneyline", `${chosen.playername}\n\nTeam ML (e.g. -297) or blank:`, ui.ButtonSet.OK_CANCEL);
  if (mlResp.getSelectedButton() !== ui.Button.OK) return;
  const mlText = mlResp.getResponseText().trim();

  const oppResp = ui.prompt("Opponent Moneyline", "Opp ML (e.g. +297) or blank:", ui.ButtonSet.OK_CANCEL);
  if (oppResp.getSelectedButton() !== ui.Button.OK) return;
  const oppText = oppResp.getResponseText().trim();

  const oppTeamResp = ui.prompt("Opponent Team (optional)", "Enter opposing team name (e.g. PCFIC) or leave blank:", ui.ButtonSet.OK_CANCEL);
  if (oppTeamResp.getSelectedButton() !== ui.Button.OK) return;
  const oppTeamText = oppTeamResp.getResponseText().trim();

  let targetRow = sheet.getLastRow() + 1;
  const selRow  = sheet.getActiveRange().getRow();
  if (selRow >= DATA_START && sheet.getRange(selRow, C.PLAYER).getValue() === "") targetRow = selRow;

  sheet.getRange(targetRow, C.PLAYER).setValue(chosen.playername);
  sheet.getRange(targetRow, C.SIDE).setValue("Blue");
  sheet.getRange(targetRow, C.FORMAT).setValue("M1-3");
  if (mlText)      sheet.getRange(targetRow, C.TEAM_ML).setValue(parseInt(mlText));
  if (oppText)     sheet.getRange(targetRow, C.OPP_ML).setValue(parseInt(oppText));
  if (oppTeamText) sheet.getRange(targetRow, C.OPP_TEAM).setValue(oppTeamText);

  try {
    const data = fetchPrediction(chosen.playername, "Blue", mlText, oppText, oppTeamText);
    writeRow(sheet, targetRow, data);
    SpreadsheetApp.getActive().toast(`✅ ${chosen.playername} done!`, "LoL Props", 4);
  } catch(e) { ui.alert("Prediction failed: " + e.message); }
}

// ─── Predict selected ─────────────────────────────────────────────────────────
function predictSelected() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const sel   = sheet.getActiveRange();
  let ok = 0, errs = [];
  for (let r = sel.getRow(); r <= sel.getLastRow(); r++) {
    if (r < DATA_START) continue;
    const inp = rowInputs(sheet, r);
    if (!inp.player) continue;
    try {
      writeRow(sheet, r, fetchPrediction(inp.player, inp.side, inp.teamML, inp.oppML, inp.oppTeam));
      ok++;
    } catch(e) { errs.push(`${inp.player}: ${e.message}`); }
  }
  SpreadsheetApp.getUi().alert(`Updated ${ok} row(s).` + (errs.length ? "\n\n"+errs.join("\n") : ""));
}

// ─── Predict all ──────────────────────────────────────────────────────────────
function predictAll() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const last  = sheet.getLastRow();
  let ok = 0, errs = [];
  for (let r = DATA_START; r <= last; r++) {
    const inp = rowInputs(sheet, r);
    if (!inp.player) continue;
    try {
      writeRow(sheet, r, fetchPrediction(inp.player, inp.side, inp.teamML, inp.oppML, inp.oppTeam));
      ok++;
      SpreadsheetApp.getActive().toast(`✅ ${inp.player}`, "LoL Props", 2);
    } catch(e) { errs.push(`Row ${r} – ${inp.player}: ${e.message}`); }
    Utilities.sleep(400);
  }
  SpreadsheetApp.getUi().alert(`Updated ${ok} row(s).` + (errs.length ? "\n\n"+errs.join("\n") : ""));
}

// ─── Refresh data ─────────────────────────────────────────────────────────────
function refreshData() {
  try {
    apiGet("/refresh");
    SpreadsheetApp.getUi().alert("✅ Data refreshed! Model reloaded latest OE data.");
  } catch(e) {
    SpreadsheetApp.getUi().alert("❌ Refresh failed:\n" + e.message);
  }
}

// ─── Setup sheet ──────────────────────────────────────────────────────────────
function setupSheet() {
  const sheet = SpreadsheetApp.getActiveSheet();
  sheet.clearContents().clearFormats();

  // Row 1: group headers
  const groups = [
    [1,6,"INPUTS (you fill)"],
    [7,9,"SPORTSBOOK LINES (you fill)"],
    [10,12,"MODEL — MAP 1"],
    [13,16,"MODEL — M1-2"],
    [17,20,"MODEL — M1-3"],
    [21,23,"EDGE (model − line)"],
    [24,28,"META"],
  ];
  const groupColours = ["#1a3a5c","#4a1a1a","#1a3a1a","#1a1a4a","#3a1a3a","#4a3a00","#2a2a2a"];
  groups.forEach(([s,e,label],i) => {
    sheet.getRange(1,s,1,e-s+1).merge().setValue(label)
      .setHorizontalAlignment("center").setFontWeight("bold")
      .setFontColor("#fff").setBackground(groupColours[i]);
  });

  // Row 2: column headers
  const headers = [
    "Player","Side","Team ML","Opp ML","Format","Opp Team",
    "SB Kills","SB Deaths","SB Assists",
    "Kills","Deaths","Assists",
    "Kills","Deaths","Assists","Fantasy",
    "Kills","Deaths","Assists","Fantasy",
    "Kills Edge","Deaths Edge","Assists Edge",
    "Position","League","Win%","Games","Recent Form","Updated"
  ];
  sheet.getRange(2,1,1,headers.length).setValues([headers])
    .setFontWeight("bold").setBackground("#222").setFontColor("#fff");

  sheet.setFrozenRows(2);
  sheet.setColumnWidth(C.PLAYER, 120);
  sheet.setColumnWidth(C.RECENT, 220);
  [C.SB_KILLS,C.SB_DEATHS,C.SB_ASSISTS].forEach(col => sheet.setColumnWidth(col, 80));
  [C.EDGE_KILLS,C.EDGE_DEATHS,C.EDGE_ASSISTS].forEach(col => {
    sheet.setColumnWidth(col, 90);
    sheet.getRange(DATA_START, col, 100, 1).setNumberFormat("0.00");
  });

  // Highlight SB line columns with a subtle yellow tint
  sheet.getRange(DATA_START, C.OPP_TEAM, 50, 1).setBackground("#f3e5f5");  // purple tint for opp team
  sheet.getRange(DATA_START, C.SB_KILLS, 50, 3).setBackground("#fffde7");

  // Example row
  sheet.getRange(DATA_START, C.PLAYER).setValue("Leave");
  sheet.getRange(DATA_START, C.SIDE).setValue("Blue");
  sheet.getRange(DATA_START, C.FORMAT).setValue("M1-3");

  SpreadsheetApp.getUi().alert(
    "Sheet ready!\n\n" +
    "WORKFLOW:\n" +
    "1. Search & predict player (or type name in col A + Predict ALL)\n" +
    "2. Type sportsbook lines in the yellow SB Kills/Deaths/Assists columns\n" +
    "3. Edge columns auto-fill:\n" +
    "   GREEN = model says OVER (good over bet)\n" +
    "   RED   = model says UNDER (good under bet)\n" +
    "   BOLD  = edge ≥ 1.5 (strong edge)\n\n" +
    "4. Change SB lines anytime → LoL Props → Recalculate edges\n\n" +
    "FORMAT column: M1, M1-2, or M1-3 (determines which model value is used for edge)"
  );
}


// ─── Best Bets Filter ────────────────────────────────────────────────────────

function showBestBets() {
  const ss         = SpreadsheetApp.getActiveSpreadsheet();
  const mainSheet  = SpreadsheetApp.getActiveSheet();
  const lastRow    = mainSheet.getLastRow();

  let betSheet = ss.getSheetByName("Best Bets");
  if (!betSheet) betSheet = ss.insertSheet("Best Bets");
  betSheet.clearContents().clearFormats();

  // Headers
  const headers = ["Player","Position","League","Format","Team ML","Opp ML","Opp Team",
                   "Stat","SB Line","Model Pred","Edge","Recommendation","Games"];
  betSheet.getRange(1,1,1,headers.length).setValues([headers])
    .setFontWeight("bold").setBackground("#1a1a2e").setFontColor("#fff");

  const rows = [];

  for (let r = DATA_START; r <= lastRow; r++) {
    const player  = mainSheet.getRange(r, C.PLAYER).getValue().toString().trim();
    if (!player) continue;

    const pos     = mainSheet.getRange(r, C.POSITION).getValue();
    const league  = mainSheet.getRange(r, C.LEAGUE).getValue();
    const format  = mainSheet.getRange(r, C.FORMAT).getValue() || "M1-3";
    const teamML  = mainSheet.getRange(r, C.TEAM_ML).getValue();
    const oppML   = mainSheet.getRange(r, C.OPP_ML).getValue();
    const oppTeam = mainSheet.getRange(r, C.OPP_TEAM).getValue();

    const stats = [
      { stat:"Kills",   sb:mainSheet.getRange(r,C.SB_KILLS).getValue(),   edge:mainSheet.getRange(r,C.EDGE_KILLS).getValue()   },
      { stat:"Deaths",  sb:mainSheet.getRange(r,C.SB_DEATHS).getValue(),  edge:mainSheet.getRange(r,C.EDGE_DEATHS).getValue()  },
      { stat:"Assists", sb:mainSheet.getRange(r,C.SB_ASSISTS).getValue(), edge:mainSheet.getRange(r,C.EDGE_ASSISTS).getValue() },
    ];

    // Get model predictions
    const fmt = (format||"M1-3").toString().toUpperCase().replace(/[\s-]/g,"");
    const modelK = fmt==="M12" ? mainSheet.getRange(r,C.M12_KILLS).getValue()   : fmt==="M1"||fmt==="MAP1" ? mainSheet.getRange(r,C.M1_KILLS).getValue()   : mainSheet.getRange(r,C.M13_KILLS).getValue();
    const modelD = fmt==="M12" ? mainSheet.getRange(r,C.M12_DEATHS).getValue()  : fmt==="M1"||fmt==="MAP1" ? mainSheet.getRange(r,C.M1_DEATHS).getValue()  : mainSheet.getRange(r,C.M13_DEATHS).getValue();
    const modelA = fmt==="M12" ? mainSheet.getRange(r,C.M12_ASSISTS).getValue() : fmt==="M1"||fmt==="MAP1" ? mainSheet.getRange(r,C.M1_ASSISTS).getValue() : mainSheet.getRange(r,C.M13_ASSISTS).getValue();
    const modelVals = { Kills:modelK, Deaths:modelD, Assists:modelA };

    for (const s of stats) {
      if (!s.sb || s.sb === "" || isNaN(parseFloat(s.sb))) continue;
      const edge = parseFloat(s.edge);
      if (isNaN(edge)) continue;

      // Skip low-sample players (< 10 games = unreliable)
      const games = mainSheet.getRange(r, C.GAMES).getValue();
      if (games > 0 && games < 10) continue;
      // Include strong picks either direction (abs edge ≥ 2.0)
      if (Math.abs(edge) < 2.0) continue;

      const rec = edge < 0 ? `🔴 UNDER ${s.sb} (edge ${edge.toFixed(1)})` : `🔵 OVER ${s.sb} (edge +${edge.toFixed(1)})`;
      rows.push([player, pos, league, format, teamML, oppML, oppTeam,
                 s.stat, s.sb, modelVals[s.stat], edge, rec, games]);
    }
  }

  if (rows.length === 0) {
    betSheet.getRange(2,1).setValue("No strong under picks found (edge ≤ -2.0)");
    ss.setActiveSheet(betSheet);
    return;
  }

  // Sort by edge (most negative first = strongest unders)
  rows.sort((a,b) => a[10] - b[10]);

  betSheet.getRange(2, 1, rows.length, headers.length).setValues(rows);

  // Colour edge column
  for (let i = 0; i < rows.length; i++) {
    const edge = rows[i][10];
    const abs  = Math.abs(edge);
    const intensity = Math.min(1, abs / 3.0);
    const r = Math.round(255 - 100 * intensity);
    const g = Math.round(200 + 55 * intensity);
    betSheet.getRange(i+2, 11).setBackground(`rgb(${r},${g},${r})`).setFontWeight("bold");
  }

  betSheet.setFrozenRows(1);
  betSheet.setColumnWidth(1, 120);
  betSheet.setColumnWidth(12, 200);
  ss.setActiveSheet(betSheet);
  SpreadsheetApp.getUi().alert(
    `✅ Found ${rows.length} strong pick(s) with |edge| ≥ 2.0\n\n` +
    `Sorted by edge strength. Both overs (🔵) and unders (🔴) hit ~75% at edge ≥ 2.\n` +
    `Recommended: 3-4 leg parlay using strongest picks.`
  );
}

// ─── Results Tracking ─────────────────────────────────────────────────────────

function setupTrackingSheet() {
  const ss    = SpreadsheetApp.getActiveSpreadsheet();
  let sheet   = ss.getSheetByName("Results Tracker");
  if (!sheet) sheet = ss.insertSheet("Results Tracker");
  sheet.clearContents().clearFormats();

  // Headers
  const headers = [
    "Date","Match","Player","Position","League",
    "Format","Team ML","Opp ML","Opp Team",
    "Stat","SB Line","Model Pred","Edge","Direction",
    "Actual Result","Hit?","Notes"
  ];
  sheet.getRange(1,1,1,headers.length).setValues([headers])
    .setFontWeight("bold").setBackground("#1a1a2e").setFontColor("#fff");
  sheet.setFrozenRows(1);

  // Column widths
  sheet.setColumnWidth(1,100);  // Date
  sheet.setColumnWidth(2,150);  // Match
  sheet.setColumnWidth(3,120);  // Player
  sheet.setColumnWidth(10,80);  // Stat
  sheet.setColumnWidth(16,80);  // Hit?
  sheet.setColumnWidth(17,200); // Notes

  // Conditional formatting for Hit column
  const hitRange = sheet.getRange(2,16,500,1);
  const greenRule = SpreadsheetApp.newConditionalFormatRule()
    .whenTextEqualTo("✅").setBackground("#c8e6c9").build();
  const redRule = SpreadsheetApp.newConditionalFormatRule()
    .whenTextEqualTo("❌").setBackground("#ffcdd2").build();
  hitRange.setConditionalFormatRules([greenRule, redRule]);

  // Edge colour: green if positive, red if negative
  const edgeRange = sheet.getRange(2,13,500,1);
  const posEdge = SpreadsheetApp.newConditionalFormatRule()
    .whenNumberGreaterThan(1.5).setBackground("#c8e6c9").setFontColor("#1b5e20").build();
  const negEdge = SpreadsheetApp.newConditionalFormatRule()
    .whenNumberLessThan(-1.5).setBackground("#ffcdd2").setFontColor("#b71c1c").build();
  edgeRange.setConditionalFormatRules([posEdge, negEdge]);

  ss.setActiveSheet(sheet);
  SpreadsheetApp.getUi().alert(
    "Results Tracker ready!\n\n" +
    "HOW TO USE:\n" +
    "1. After running predictions, select player rows on main sheet\n" +
    "2. LoL Props → Log selected predictions to tracker\n" +
    "3. After games finish, fill in \'Actual Result\' column\n" +
    "4. LoL Props → Update hit/miss → auto-fills the Hit? column\n\n" +
    "COLUMNS:\n" +
    "• SB Line = sportsbook line you entered\n" +
    "• Model Pred = what the model predicted\n" +
    "• Edge = model - line (positive = over, negative = under)\n" +
    "• Direction = OVER or UNDER (what the model recommends)\n" +
    "• Actual Result = what actually happened (you fill this)\n" +
    "• Hit? = ✅ if model was right, ❌ if wrong (auto-filled)"
  );
}


function logSelectedPredictions() {
  const ss       = SpreadsheetApp.getActiveSpreadsheet();
  const mainSheet = SpreadsheetApp.getActiveSheet();
  const sel      = mainSheet.getActiveRange();

  let trackSheet = ss.getSheetByName("Results Tracker");
  if (!trackSheet) {
    setupTrackingSheet();
    trackSheet = ss.getSheetByName("Results Tracker");
  }

  const today   = new Date().toLocaleDateString();
  let logged    = 0;
  const rows    = [];

  for (let r = sel.getRow(); r <= sel.getLastRow(); r++) {
    if (r < DATA_START) continue;
    const player  = mainSheet.getRange(r, C.PLAYER).getValue().toString().trim();
    if (!player) continue;

    const teamML  = mainSheet.getRange(r, C.TEAM_ML).getValue();
    const oppML   = mainSheet.getRange(r, C.OPP_ML).getValue();
    const format  = mainSheet.getRange(r, C.FORMAT).getValue() || "M1-3";
    const oppTeam = mainSheet.getRange(r, C.OPP_TEAM).getValue();
    const pos     = mainSheet.getRange(r, C.POSITION).getValue();
    const league  = mainSheet.getRange(r, C.LEAGUE).getValue();

    const sbKills   = mainSheet.getRange(r, C.SB_KILLS).getValue();
    const sbDeaths  = mainSheet.getRange(r, C.SB_DEATHS).getValue();
    const sbAssists = mainSheet.getRange(r, C.SB_ASSISTS).getValue();

    const edgeKills   = mainSheet.getRange(r, C.EDGE_KILLS).getValue();
    const edgeDeaths  = mainSheet.getRange(r, C.EDGE_DEATHS).getValue();
    const edgeAssists = mainSheet.getRange(r, C.EDGE_ASSISTS).getValue();

    // Get model predictions based on format
    const fmt = (format||"M1-3").toString().toUpperCase().replace(/[\s-]/g,"");
    let mkills, mdeaths, massists;
    if (fmt === "M1" || fmt === "MAP1") {
      mkills   = mainSheet.getRange(r, C.M1_KILLS).getValue();
      mdeaths  = mainSheet.getRange(r, C.M1_DEATHS).getValue();
      massists = mainSheet.getRange(r, C.M1_ASSISTS).getValue();
    } else if (fmt === "M12") {
      mkills   = mainSheet.getRange(r, C.M12_KILLS).getValue();
      mdeaths  = mainSheet.getRange(r, C.M12_DEATHS).getValue();
      massists = mainSheet.getRange(r, C.M12_ASSISTS).getValue();
    } else {
      mkills   = mainSheet.getRange(r, C.M13_KILLS).getValue();
      mdeaths  = mainSheet.getRange(r, C.M13_DEATHS).getValue();
      massists = mainSheet.getRange(r, C.M13_ASSISTS).getValue();
    }

    const matchName = `${player} vs ${oppTeam || "?"}`;

    // Log kills if SB line exists
    if (sbKills !== "" && sbKills !== null && !isNaN(parseFloat(sbKills))) {
      const edge = parseFloat(edgeKills) || 0;
      rows.push([today, matchName, player, pos, league, format,
                 teamML, oppML, oppTeam, "Kills",
                 sbKills, mkills, edge,
                 edge >= 0 ? "OVER" : "UNDER",
                 "", "", ""]);
      logged++;
    }
    // Log deaths
    if (sbDeaths !== "" && sbDeaths !== null && !isNaN(parseFloat(sbDeaths))) {
      const edge = parseFloat(edgeDeaths) || 0;
      rows.push([today, matchName, player, pos, league, format,
                 teamML, oppML, oppTeam, "Deaths",
                 sbDeaths, mdeaths, edge,
                 edge >= 0 ? "OVER" : "UNDER",
                 "", "", ""]);
      logged++;
    }
    // Log assists
    if (sbAssists !== "" && sbAssists !== null && !isNaN(parseFloat(sbAssists))) {
      const edge = parseFloat(edgeAssists) || 0;
      rows.push([today, matchName, player, pos, league, format,
                 teamML, oppML, oppTeam, "Assists",
                 sbAssists, massists, edge,
                 edge >= 0 ? "OVER" : "UNDER",
                 "", "", ""]);
      logged++;
    }
  }

  if (rows.length === 0) {
    SpreadsheetApp.getUi().alert("No predictions with SB lines found in selected rows.");
    return;
  }

  const lastRow = Math.max(trackSheet.getLastRow(), 1);
  trackSheet.getRange(lastRow + 1, 1, rows.length, rows[0].length).setValues(rows);
  ss.setActiveSheet(trackSheet);
  SpreadsheetApp.getUi().alert(`✅ Logged ${logged} prediction(s) to Results Tracker.`);
}


function updateHitMiss() {
  const ss         = SpreadsheetApp.getActiveSpreadsheet();
  const trackSheet = ss.getSheetByName("Results Tracker");
  if (!trackSheet) { SpreadsheetApp.getUi().alert("No Results Tracker found. Run Setup Tracking Sheet first."); return; }

  const lastRow = trackSheet.getLastRow();
  if (lastRow < 2) { SpreadsheetApp.getUi().alert("No data in tracker yet."); return; }

  let updated = 0;
  for (let r = 2; r <= lastRow; r++) {
    const edge    = parseFloat(trackSheet.getRange(r, 13).getValue());  // Edge col
    const dir     = trackSheet.getRange(r, 14).getValue().toString();   // Direction
    const actual  = parseFloat(trackSheet.getRange(r, 15).getValue());  // Actual Result
    const sbLine  = parseFloat(trackSheet.getRange(r, 11).getValue());  // SB Line
    const hitCell = trackSheet.getRange(r, 16);                         // Hit?

    if (isNaN(actual) || actual === 0 && trackSheet.getRange(r,15).getValue() === "") continue;
    if (isNaN(sbLine)) continue;

    const wentOver = actual > sbLine;
    const modelSaysOver = dir === "OVER";
    const hit = wentOver === modelSaysOver;

    hitCell.setValue(hit ? "✅" : "❌");
    updated++;
  }

  // Summary stats
  const allHits = trackSheet.getRange(2, 16, lastRow - 1, 1).getValues().flat();
  const wins    = allHits.filter(v => v === "✅").length;
  const losses  = allHits.filter(v => v === "❌").length;
  const total   = wins + losses;
  const pct     = total > 0 ? (wins/total*100).toFixed(1) : "–";

  SpreadsheetApp.getUi().alert(
    `Updated ${updated} result(s).\n\n` +
    `Overall Record: ${wins}W - ${losses}L (${pct}% hit rate)\n\n` +
    (total >= 20 ? (parseFloat(pct) > 55 ? "📈 Model showing positive edge!" : "📊 Need more data or model adjustment.") : `Collect ${20-total} more results for a reliable sample.`)
  );
}


function getModelAccuracySummary() {
  const ss         = SpreadsheetApp.getActiveSpreadsheet();
  const trackSheet = ss.getSheetByName("Results Tracker");
  if (!trackSheet) { SpreadsheetApp.getUi().alert("No Results Tracker found."); return; }

  const lastRow = trackSheet.getLastRow();
  if (lastRow < 2) { SpreadsheetApp.getUi().alert("No data yet."); return; }

  const data = trackSheet.getRange(2, 1, lastRow-1, 17).getValues();

  // Group by stat and edge size
  const stats    = { Kills:{w:0,l:0}, Deaths:{w:0,l:0}, Assists:{w:0,l:0} };
  const byEdge   = { "0-1":{w:0,l:0}, "1-2":{w:0,l:0}, "2+":{w:0,l:0} };
  const byLeague = {};
  const byPos    = {};
  const byPosEdge = {};
  const byDir    = { OVER:{w:0,l:0}, UNDER:{w:0,l:0} };
  const byLeaguePos = {}; // league → position → {w,l}

  for (const row of data) {
    const stat   = row[9];
    const edge   = Math.abs(parseFloat(row[12]));
    const hit    = row[15];
    const league = row[4];
    const pos    = row[3];
    const dir    = row[13].toString();

    if (hit !== "✅" && hit !== "❌") continue;
    const win = hit === "✅";

    // By stat
    if (stats[stat]) { win ? stats[stat].w++ : stats[stat].l++; }

    // By edge bucket
    const bucket = edge < 1 ? "0-1" : edge < 2 ? "1-2" : "2+";
    win ? byEdge[bucket].w++ : byEdge[bucket].l++;

    // By league
    if (!byLeague[league]) byLeague[league] = {w:0,l:0};
    win ? byLeague[league].w++ : byLeague[league].l++;

    // By position
    if (!byPos[pos]) byPos[pos] = {w:0,l:0};
    win ? byPos[pos].w++ : byPos[pos].l++;

    // By position (strong edges only)
    if (edge >= 1.5) {
      if (!byPosEdge[pos]) byPosEdge[pos] = {w:0,l:0};
      win ? byPosEdge[pos].w++ : byPosEdge[pos].l++;
    }

    // By direction
    if (byDir[dir]) win ? byDir[dir].w++ : byDir[dir].l++;

    // By league + position
    if (!byLeaguePos[league]) byLeaguePos[league] = {};
    if (!byLeaguePos[league][pos]) byLeaguePos[league][pos] = {w:0,l:0};
    win ? byLeaguePos[league][pos].w++ : byLeaguePos[league][pos].l++;
  }

  const fmt = (obj) => {
    const t = obj.w + obj.l;
    return t === 0 ? "–" : `${obj.w}W-${obj.l}L (${(obj.w/t*100).toFixed(0)}%)`;
  };

  const posOrder = ["top","jng","mid","bot","sup"];

  // Write to a summary sheet instead of alert (too much data for a dialog)
  let summarySheet = ss.getSheetByName("Accuracy Summary");
  if (!summarySheet) summarySheet = ss.insertSheet("Accuracy Summary");
  summarySheet.clearContents().clearFormats();

  const rows = [];
  rows.push(["=== MODEL ACCURACY SUMMARY ===", "", ""]);
  rows.push(["", "", ""]);

  rows.push(["BY STAT", "Record", "Hit%"]);
  for (const [s,v] of Object.entries(stats)) {
    const t = v.w+v.l;
    rows.push([s, `${v.w}W-${v.l}L`, t > 0 ? `${(v.w/t*100).toFixed(0)}%` : "–"]);
  }
  rows.push(["","",""]);

  rows.push(["BY POSITION", "Record", "Hit%"]);
  for (const p of posOrder) {
    if (!byPos[p]) continue;
    const v = byPos[p]; const t = v.w+v.l;
    rows.push([p, `${v.w}W-${v.l}L`, t > 0 ? `${(v.w/t*100).toFixed(0)}%` : "–"]);
  }
  rows.push(["","",""]);

  rows.push(["BY POSITION (edge ≥1.5)", "Record", "Hit%"]);
  for (const p of posOrder) {
    if (!byPosEdge[p]) continue;
    const v = byPosEdge[p]; const t = v.w+v.l;
    rows.push([p, `${v.w}W-${v.l}L`, t > 0 ? `${(v.w/t*100).toFixed(0)}%` : "–"]);
  }
  rows.push(["","",""]);

  rows.push(["OVER vs UNDER", "Record", "Hit%"]);
  for (const [d,v] of Object.entries(byDir)) {
    const t = v.w+v.l;
    rows.push([d, `${v.w}W-${v.l}L`, t > 0 ? `${(v.w/t*100).toFixed(0)}%` : "–"]);
  }
  rows.push(["","",""]);

  rows.push(["BY EDGE SIZE", "Record", "Hit%"]);
  for (const [b,v] of Object.entries(byEdge)) {
    const t = v.w+v.l;
    rows.push([`Edge ${b}`, `${v.w}W-${v.l}L`, t > 0 ? `${(v.w/t*100).toFixed(0)}%` : "–"]);
  }
  rows.push(["","",""]);

  rows.push(["BY LEAGUE", "Record", "Hit%"]);
  for (const [l,v] of Object.entries(byLeague)) {
    const t = v.w+v.l;
    rows.push([l, `${v.w}W-${v.l}L`, t > 0 ? `${(v.w/t*100).toFixed(0)}%` : "–"]);
  }
  rows.push(["","",""]);

  rows.push(["BY LEAGUE + POSITION", "Record", "Hit%"]);
  for (const league of Object.keys(byLeaguePos).sort()) {
    for (const p of posOrder) {
      const v = byLeaguePos[league][p];
      if (!v) continue;
      const t = v.w+v.l;
      if (t < 3) continue; // Skip tiny samples
      rows.push([`${league} — ${p}`, `${v.w}W-${v.l}L`, t > 0 ? `${(v.w/t*100).toFixed(0)}%` : "–"]);
    }
  }

  summarySheet.getRange(1, 1, rows.length, 3).setValues(rows);

  // Style headers
  summarySheet.getRange(1,1).setFontWeight("bold").setFontSize(13);
  const headerRows = [3, 9, 14, 19, 23, 28, 33];
  for (const r of headerRows) {
    try { summarySheet.getRange(r,1,1,3).setFontWeight("bold").setBackground("#1a1a2e").setFontColor("#fff"); } catch(e) {}
  }

  // Colour hit% column: green if ≥60%, red if <45%
  for (let r = 1; r <= rows.length; r++) {
    const cell = summarySheet.getRange(r, 3);
    const val  = cell.getValue().toString();
    if (!val.includes("%")) continue;
    const pctNum = parseFloat(val);
    if (pctNum >= 60) cell.setBackground("#c8e6c9");
    else if (pctNum < 45) cell.setBackground("#ffcdd2");
  }

  summarySheet.setColumnWidth(1, 200);
  summarySheet.setColumnWidth(2, 120);
  summarySheet.setColumnWidth(3, 80);
  ss.setActiveSheet(summarySheet);
  SpreadsheetApp.getUi().alert("✅ Accuracy Summary updated! Check the 'Accuracy Summary' tab.");
}

// ─── Status ───────────────────────────────────────────────────────────────────
function checkStatus() {
  try {
    const d = apiGet("/");
    SpreadsheetApp.getUi().alert(`✅ API Online\nPlayers: ${d.players}\nDate range: ${d.date_range}`);
  } catch(e) {
    SpreadsheetApp.getUi().alert(`❌ Can't reach API:\n${e.message}`);
  }
}
