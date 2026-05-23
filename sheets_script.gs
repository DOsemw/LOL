/**
 * LoL Player Props — Google Sheets Script
 * ========================================
 * HOW TO INSTALL:
 *   1. Open your Google Sheet
 *   2. Extensions → Apps Script
 *   3. Delete everything, paste this entire file
 *   4. Change API_URL on line 10 to your Railway URL
 *   5. Ctrl+S to save, close tab, reload sheet
 *   6. Click "LoL Props" menu → Setup Sheet
 */

const API_URL = "https://YOUR-APP.up.railway.app"; // ← CHANGE THIS

// ─── Column layout ────────────────────────────────────────────────────────────
// Inputs (you fill):  A=Player  B=Side  C=Team ML  D=Opp ML
// Outputs (auto):     E=Position  F=League  G=Win%
//   M1 (single map):  H=Kills  I=Deaths  J=Assists
//   M1-2 (2 maps):    K=Kills  L=Deaths  M=Assists  N=Fantasy
//   M1-3 (full Bo3):  O=Kills  P=Deaths  Q=Assists  R=Fantasy
//   Meta:             S=Recent Form  T=Updated

const C = {
  PLAYER:1, SIDE:2, TEAM_ML:3, OPP_ML:4,
  POSITION:5, LEAGUE:6, WIN_PCT:7,
  M1_KILLS:8,  M1_DEATHS:9,  M1_ASSISTS:10,
  M12_KILLS:11, M12_DEATHS:12, M12_ASSISTS:13, M12_FANTASY:14,
  M13_KILLS:15, M13_DEATHS:16, M13_ASSISTS:17, M13_FANTASY:18,
  RECENT:19, UPDATED:20
};

// ─── Menu ─────────────────────────────────────────────────────────────────────
function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu("LoL Props")
    .addItem("🔍 Search & predict player",   "searchAndPredict")
    .addItem("▶  Predict selected rows",     "predictSelected")
    .addItem("▶  Predict ALL rows",          "predictAll")
    .addSeparator()
    .addItem("📋 Setup sheet",               "setupSheet")
    .addItem("🩺 Check API status",          "checkStatus")
    .addToUi();
}

// ─── API call ─────────────────────────────────────────────────────────────────
function apiGet(path) {
  const resp = UrlFetchApp.fetch(API_URL + path, {
    method: "GET", muteHttpExceptions: true,
    headers: { Accept: "application/json" }
  });
  if (resp.getResponseCode() !== 200) {
    const body = JSON.parse(resp.getContentText());
    const msg  = body.detail?.message || body.detail || `HTTP ${resp.getResponseCode()}`;
    throw new Error(msg);
  }
  return JSON.parse(resp.getContentText());
}

// ─── Search & predict (dialog) ────────────────────────────────────────────────
function searchAndPredict() {
  const ui = SpreadsheetApp.getUi();
  const sheet = SpreadsheetApp.getActiveSheet();

  // Ask for player name
  const nameResp = ui.prompt("LoL Props — Search Player",
    "Type part of a player name (e.g. 'fak' for Faker):", ui.ButtonSet.OK_CANCEL);
  if (nameResp.getSelectedButton() !== ui.Button.OK) return;
  const query = nameResp.getResponseText().trim();
  if (!query) return;

  // Search API
  let results;
  try { results = apiGet(`/search?q=${encodeURIComponent(query)}`); }
  catch(e) { ui.alert("Search failed: " + e.message); return; }

  if (!results.length) { ui.alert("No players found matching: " + query); return; }

  // Show results as numbered list
  const listText = results.map((p, i) =>
    `${i+1}. ${p.playername} — ${p.position} — ${p.league} (${p.games} games)`
  ).join("\n");

  const pickResp = ui.prompt("LoL Props — Pick a Player",
    `Results:\n${listText}\n\nEnter the number to select:`, ui.ButtonSet.OK_CANCEL);
  if (pickResp.getSelectedButton() !== ui.Button.OK) return;

  const pick = parseInt(pickResp.getResponseText().trim()) - 1;
  if (isNaN(pick) || pick < 0 || pick >= results.length) {
    ui.alert("Invalid selection."); return;
  }
  const chosen = results[pick];

  // Ask for moneyline
  const mlResp = ui.prompt("LoL Props — Moneyline (optional)",
    `Player: ${chosen.playername}\n\nEnter team moneyline (e.g. -297) or leave blank for 50/50:`,
    ui.ButtonSet.OK_CANCEL);
  if (mlResp.getSelectedButton() !== ui.Button.OK) return;
  const mlText = mlResp.getResponseText().trim();

  const oppResp = ui.prompt("LoL Props — Opponent Moneyline (optional)",
    `Enter opponent moneyline (e.g. +297) or leave blank:`, ui.ButtonSet.OK_CANCEL);
  if (oppResp.getSelectedButton() !== ui.Button.OK) return;
  const oppText = oppResp.getResponseText().trim();

  // Find next empty row
  const lastRow = Math.max(sheet.getLastRow(), 1);
  let targetRow = lastRow + 1;
  // Check if current selected row is empty
  const selRow = sheet.getActiveRange().getRow();
  if (selRow >= 2 && sheet.getRange(selRow, C.PLAYER).getValue() === "") {
    targetRow = selRow;
  }

  // Write inputs
  sheet.getRange(targetRow, C.PLAYER).setValue(chosen.playername);
  sheet.getRange(targetRow, C.SIDE).setValue("Blue");
  if (mlText)  sheet.getRange(targetRow, C.TEAM_ML).setValue(parseInt(mlText));
  if (oppText) sheet.getRange(targetRow, C.OPP_ML).setValue(parseInt(oppText));

  // Run prediction
  try {
    const data = fetchPrediction(chosen.playername, "Blue", mlText, oppText);
    writeRow(sheet, targetRow, data);
    SpreadsheetApp.getActive().toast(`✅ ${chosen.playername} predicted!`, "LoL Props", 4);
  } catch(e) {
    ui.alert("Prediction failed: " + e.message);
  }
}

// ─── Fetch prediction from API ────────────────────────────────────────────────
function fetchPrediction(player, side, teamML, oppML) {
  let path = `/predict?player=${encodeURIComponent(player)}&side=${encodeURIComponent(side||"Blue")}`;
  const ml  = parseInt(teamML);
  const oml = parseInt(oppML);
  if (!isNaN(ml) && !isNaN(oml)) path += `&moneyline=${ml}&opp_ml=${oml}`;
  return apiGet(path);
}

// ─── Write one row of results ─────────────────────────────────────────────────
function writeRow(sheet, row, d) {
  const m1  = d.map1  || {};
  const m12 = d.m1_2  || {};
  const m13 = d.m1_3  || {};

  const recentStr = (d.recent_form||[]).slice(0,3)
    .map(g => `${g.champion} ${g.kills}/${g.deaths}/${g.assists}`).join(", ");

  const winPct = d.win_prob ? (d.win_prob*100).toFixed(0)+"%" : "–";

  // Info columns
  sheet.getRange(row, C.POSITION).setValue(d.position || "–");
  sheet.getRange(row, C.LEAGUE).setValue(d.league || "–");
  sheet.getRange(row, C.WIN_PCT).setValue(winPct);

  // M1 (single map)
  sheet.getRange(row, C.M1_KILLS).setValue(m1.kills?.expected ?? "–");
  sheet.getRange(row, C.M1_DEATHS).setValue(m1.deaths?.expected ?? "–");
  sheet.getRange(row, C.M1_ASSISTS).setValue(m1.assists?.expected ?? "–");

  // M1-2
  sheet.getRange(row, C.M12_KILLS).setValue(m12.kills?.series_total ?? "–");
  sheet.getRange(row, C.M12_DEATHS).setValue(m12.deaths?.series_total ?? "–");
  sheet.getRange(row, C.M12_ASSISTS).setValue(m12.assists?.series_total ?? "–");
  sheet.getRange(row, C.M12_FANTASY).setValue(m12.fantasy ?? "–");

  // M1-3
  sheet.getRange(row, C.M13_KILLS).setValue(m13.kills?.series_total ?? "–");
  sheet.getRange(row, C.M13_DEATHS).setValue(m13.deaths?.series_total ?? "–");
  sheet.getRange(row, C.M13_ASSISTS).setValue(m13.assists?.series_total ?? "–");
  sheet.getRange(row, C.M13_FANTASY).setValue(m13.fantasy ?? "–");

  sheet.getRange(row, C.RECENT).setValue(recentStr);
  sheet.getRange(row, C.UPDATED).setValue(new Date().toLocaleString());

  // Colour kills green↑ red↓
  if (m13.kills) {
    colour(sheet.getRange(row, C.M13_KILLS),   m13.kills.series_total,   0, 20);
    colour(sheet.getRange(row, C.M13_ASSISTS), m13.assists.series_total, 0, 35);
    colour(sheet.getRange(row, C.M13_FANTASY), m13.fantasy||0,           20, 100);
    colour(sheet.getRange(row, C.M12_KILLS),   m12.kills.series_total,   0, 14);
    colour(sheet.getRange(row, C.M12_ASSISTS), m12.assists.series_total, 0, 25);
  }
}

function colour(range, val, min, max) {
  const pct = Math.min(1, Math.max(0, (val-min)/(max-min)));
  range.setBackground(`rgb(${Math.round(255*(1-pct))},${Math.round(200*pct)},60)`);
}

// ─── Read inputs from a row ───────────────────────────────────────────────────
function rowInputs(sheet, row) {
  return {
    player: sheet.getRange(row, C.PLAYER).getValue().toString().trim(),
    side:   sheet.getRange(row, C.SIDE).getValue().toString().trim() || "Blue",
    teamML: sheet.getRange(row, C.TEAM_ML).getValue(),
    oppML:  sheet.getRange(row, C.OPP_ML).getValue(),
  };
}

// ─── Predict selected rows ────────────────────────────────────────────────────
function predictSelected() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const sel   = sheet.getActiveRange();
  let ok = 0, errs = [];
  for (let r = sel.getRow(); r <= sel.getLastRow(); r++) {
    if (r < 2) continue;
    const inp = rowInputs(sheet, r);
    if (!inp.player) continue;
    try {
      writeRow(sheet, r, fetchPrediction(inp.player, inp.side, inp.teamML, inp.oppML));
      ok++;
    } catch(e) { errs.push(`${inp.player}: ${e.message}`); }
  }
  SpreadsheetApp.getUi().alert(`Updated ${ok} row(s).` + (errs.length ? "\n\n"+errs.join("\n") : ""));
}

// ─── Predict all rows ─────────────────────────────────────────────────────────
function predictAll() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const last  = sheet.getLastRow();
  let ok = 0, errs = [];
  for (let r = 2; r <= last; r++) {
    const inp = rowInputs(sheet, r);
    if (!inp.player) continue;
    try {
      writeRow(sheet, r, fetchPrediction(inp.player, inp.side, inp.teamML, inp.oppML));
      ok++;
      SpreadsheetApp.getActive().toast(`✅ ${inp.player}`, "LoL Props", 2);
    } catch(e) { errs.push(`Row ${r} – ${inp.player}: ${e.message}`); }
    Utilities.sleep(400);
  }
  SpreadsheetApp.getUi().alert(`Updated ${ok} row(s).` + (errs.length ? "\n\n"+errs.join("\n") : ""));
}

// ─── Setup sheet ─────────────────────────────────────────────────────────────
function setupSheet() {
  const sheet = SpreadsheetApp.getActiveSheet();
  sheet.clearContents().clearFormats();

  // Row 1: group headers
  const groups = [
    [1,4,"INPUTS (you fill)"],
    [5,7,"INFO"],
    [8,10,"MAP 1 (single map)"],
    [11,14,"M1-2 (maps 1+2)"],
    [15,18,"M1-3 (full Bo3)"],
    [19,20,"META"],
  ];
  groups.forEach(([start, end, label]) => {
    const r = sheet.getRange(1, start, 1, end-start+1);
    r.merge().setValue(label).setHorizontalAlignment("center").setFontWeight("bold");
  });
  sheet.getRange(1,1,1,4).setBackground("#1a3a5c").setFontColor("#fff");
  sheet.getRange(1,5,1,3).setBackground("#2d4a2d").setFontColor("#fff");
  sheet.getRange(1,8,1,3).setBackground("#3a2d1a").setFontColor("#fff");
  sheet.getRange(1,11,1,4).setBackground("#1a1a4a").setFontColor("#fff");
  sheet.getRange(1,15,1,4).setBackground("#3a1a3a").setFontColor("#fff");
  sheet.getRange(1,19,1,2).setBackground("#2a2a2a").setFontColor("#fff");

  // Row 2: column headers
  const headers = [
    "Player","Side","Team ML","Opp ML",
    "Position","League","Win%",
    "Kills","Deaths","Assists",
    "Kills","Deaths","Assists","Fantasy",
    "Kills","Deaths","Assists","Fantasy",
    "Recent Form (last 3)","Updated"
  ];
  sheet.getRange(2,1,1,headers.length).setValues([headers])
    .setFontWeight("bold").setBackground("#333").setFontColor("#fff");

  sheet.setFrozenRows(2);
  sheet.setColumnWidth(C.PLAYER, 120);
  sheet.setColumnWidth(C.RECENT, 220);

  // Example row
  sheet.getRange(3,C.PLAYER).setValue("Faker");
  sheet.getRange(3,C.SIDE).setValue("Blue");
  sheet.getRange(3,C.TEAM_ML).setValue(-297);
  sheet.getRange(3,C.OPP_ML).setValue(297);

  SpreadsheetApp.getUi().alert(
    "Sheet ready!\n\n" +
    "HOW TO USE:\n" +
    "• LoL Props → Search & predict player  (easiest — search by name)\n" +
    "• Or type player names in column A, then Predict ALL rows\n\n" +
    "COLUMNS:\n" +
    "• Team ML / Opp ML = American odds (e.g. -297 / +297)\n" +
    "• M1-2 = kills/deaths/assists across maps 1 and 2\n" +
    "• M1-3 = kills/deaths/assists across full Bo3\n" +
    "• Fantasy = kills×3 + assists×1.5 − deaths"
  );
}

// ─── Status check ─────────────────────────────────────────────────────────────
function checkStatus() {
  try {
    const d = apiGet("/");
    SpreadsheetApp.getUi().alert(
      `✅ API Status: ${d.status}\nPlayers in dataset: ${d.players}\nDate range: ${d.date_range}`
    );
  } catch(e) {
    SpreadsheetApp.getUi().alert(`❌ Can't reach API:\n${e.message}\n\nMake sure API_URL on line 10 is set correctly.`);
  }
}
