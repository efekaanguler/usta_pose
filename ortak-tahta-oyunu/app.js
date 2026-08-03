const SHAPE_ICONS = { Daire: "●", Kare: "■", Üçgen: "▲", Altıgen: "⬢", Kalp: "♥", Yıldız: "★" };
const COLOR_VALUES = {
  Sarı: "#e7b72c", Yeşil: "#49a645", Siyah: "#263743", Turuncu: "#ef841f",
  Kırmızı: "#dc4c43", Lacivert: "#285681", Gri: "#aeb3b6", Beyaz: "#f8f8f3", Mavi: "#237cb0"
};
const LIGHT_COLORS = new Set(["Sarı", "Gri", "Beyaz"]);

const DISKS = [
  ["yellow-1", "Sarı", 1, "Daire"], ["yellow-4", "Sarı", 4, "Yıldız"],
  ["green-2", "Yeşil", 2, "Altıgen"], ["green-5", "Yeşil", 5, "Kare"],
  ["black-3", "Siyah", 3, "Üçgen"], ["black-6", "Siyah", 6, "Kalp"],
  ["orange-1", "Turuncu", 1, "Kare"], ["orange-5", "Turuncu", 5, "Daire"],
  ["red-2", "Kırmızı", 2, "Kalp"], ["red-6", "Kırmızı", 6, "Yıldız"],
  ["navy-3", "Lacivert", 3, "Daire"], ["navy-4", "Lacivert", 4, "Altıgen"],
  ["gray-1", "Gri", 1, "Üçgen"], ["gray-6", "Gri", 6, "Altıgen"],
  ["white-2", "Beyaz", 2, "Yıldız"], ["white-4", "Beyaz", 4, "Kalp"],
  ["blue-3", "Mavi", 3, "Kare"], ["blue-5", "Mavi", 5, "Üçgen"]
].map(([id, color, number, shape]) => ({ id, color, number, shape }));

const DISK_MAP = Object.fromEntries(DISKS.map(disk => [disk.id, disk]));
const SCENARIO_ONE_P1 = ["yellow-4", "green-2", "black-3", "black-6", "red-6", "navy-4", "gray-6", "white-2", "blue-5"];
const SCENARIO_ONE_P2 = DISKS.map(disk => disk.id).filter(id => !SCENARIO_ONE_P1.includes(id));
const BOARD_NUMBERS = [
  [1, 2, 3, 4, 5, 6],
  [6, 5, 4, 3, 2, 1],
  [1, 2, 3, 4, 5, 6]
];
const BOARD_TYPES = [
  ["A", "B", "A", "N", "B", "N"],
  ["B", "N", "B", "A", "N", "A"],
  ["N", "A", "N", "B", "A", "B"]
];
const SPECIAL_CELLS = {
  8: { type: "B" },
  9: { type: "A" },
  16: { type: "A" },
  17: { type: "B" }
};
const TARGETS = {
  circles: { cell: "A", primary: "Daire", primaryPoints: 4, secondary: "Kare", secondaryPoints: 3 },
  polygons: { cell: "B", primary: "Altıgen", primaryPoints: 4, secondary: "Yıldız", secondaryPoints: 3 }
};
const STORAGE_KEY = "usta-zincirli-ortak-tahta-v3";

let state = null;
let setupScenario = 1;
let tradeSelection = { P1: null, P2: null };
let toastTimer = null;
let audioContext = null;
let physicalScoreState = {
  initialized: false,
  scenario: 1,
  names: { P1: "Oyuncu 1", P2: "Oyuncu 2" },
  shapes: Array(18).fill(null),
  detectionConfidence: Array(18).fill(null),
  detections: [],
  selectedCell: null,
  photoUrl: null,
  rotation: 0,
  analyzing: false
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

function initialState(scenario, names) {
  const hands = scenario === 1
    ? { P1: [...SCENARIO_ONE_P1], P2: [...SCENARIO_ONE_P2] }
    : { P1: [...SCENARIO_ONE_P2], P2: [...SCENARIO_ONE_P1] };
  const targets = scenario === 1
    ? { P1: "circles", P2: "polygons" }
    : { P1: "polygons", P2: "circles" };
  return {
    scenario,
    names: { P1: names.P1 || "Oyuncu 1", P2: names.P2 || "Oyuncu 2" },
    activePlayer: "P1",
    selectedDisk: null,
    hands,
    targets,
    board: Array(18).fill(null),
    trades: 0,
    history: [],
    sound: true,
    startedAt: Date.now()
  };
}

function saveState() {
  if (!state) return;
  const safeState = { ...state, selectedDisk: null };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(safeState));
}

function loadSavedState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved?.board?.length === 18 && saved?.hands?.P1 && saved?.hands?.P2) return saved;
  } catch (_) {}
  return null;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function targetHtml(targetKey, player) {
  const target = TARGETS[targetKey];
  const cells = target.cell === "A" ? "3A ve 5A" : "4B ve 6B";
  return `<small>GİZLİ PUAN DEĞERLERİ</small><strong><b>${target.cell}</b> hücresindeki her ${target.primary}: +${target.primaryPoints}</strong><strong><b>${target.cell}</b> hücresindeki her ${target.secondary}: +${target.secondaryPoints}</strong><small>${cells} yalnızca ${player} tarafından tamamlanır.</small>`;
}

function specialOwner(type) {
  return Object.keys(state.targets).find(player => TARGETS[state.targets[player]].cell === type);
}

function cellNumber(index) {
  const row = Math.floor(index / 6);
  return BOARD_NUMBERS[row][index % 6];
}

function consecutivePlacementCount(player) {
  let count = 0;
  for (let index = state.history.length - 1; index >= 0; index -= 1) {
    const action = state.history[index];
    if (action.type !== "place") continue;
    if (action.player !== player) break;
    count += 1;
  }
  return count;
}

function placementIssue(disk, cellIndex, player) {
  const row = Math.floor(cellIndex / 6);
  const number = cellNumber(cellIndex);
  if (state.board[cellIndex]) return "Bu hücre zaten dolu.";
  if (consecutivePlacementCount(player) >= 2) {
    const otherPlayer = player === "P1" ? "P2" : "P1";
    return `${state.names[player]} üst üste iki disk koydu; şimdi ${state.names[otherPlayer]} bir disk koymalı.`;
  }
  if (disk.number !== number) return `Bu hücre ${number} numaralı bir disk bekliyor.`;
  const rowStart = row * 6;
  const firstGap = state.board.slice(rowStart, rowStart + 6).findIndex(placement => !placement);
  if (firstGap !== cellIndex % 6) return `Bu zincirde önce ${cellNumber(rowStart + firstGap)} numaralı hücre tamamlanmalı.`;

  const special = SPECIAL_CELLS[cellIndex];
  if (special) {
    const owner = specialOwner(special.type);
    if (owner !== player) return `${number}${special.type} hücresini yalnızca ${state.names[owner]} tamamlayabilir.`;
    return null;
  }
  return null;
}

function diskHtml(disk, extraClass = "") {
  const light = LIGHT_COLORS.has(disk.color) ? " light" : "";
  return `<button class="disk-button${light} ${extraClass}" data-disk="${disk.id}" type="button" style="background:${COLOR_VALUES[disk.color]}" aria-label="${disk.color}, ${disk.number}, ${disk.shape}"><span class="disk-number">${disk.number}</span><span class="disk-shape" aria-hidden="true">${SHAPE_ICONS[disk.shape]}</span></button>`;
}

function renderBoard() {
  const board = $("#board");
  board.innerHTML = state.board.map((placement, index) => {
    const row = Math.floor(index / 6);
    const column = index % 6;
    const number = cellNumber(index);
    const type = BOARD_TYPES[row][column];
    const selected = state.selectedDisk ? DISK_MAP[state.selectedDisk] : null;
    const issue = selected ? placementIssue(selected, index, state.activePlayer) : null;
    const eligible = selected && !issue;
    const rowFirstGap = state.board.slice(row * 6, row * 6 + 6).findIndex(item => !item);
    const chainLocked = !placement && rowFirstGap !== column;
    const special = SPECIAL_CELLS[index];
    let content = "";
    if (placement) {
      const disk = DISK_MAP[placement.diskId];
      const light = LIGHT_COLORS.has(disk.color) ? " light" : "";
      content = `<div class="placed-disk${light}" style="background:${COLOR_VALUES[disk.color]}" title="${disk.color} · ${disk.number}/${disk.shape}"><strong>${disk.number}</strong><small>${SHAPE_ICONS[disk.shape]}</small></div>`;
    }
    const owner = special ? specialOwner(type) : null;
    const specialInfo = special ? `<span class="special-seal ${owner.toLowerCase()}">Ö${owner}</span>` : "";
    return `<button class="board-cell type-${type}${eligible ? " eligible" : ""}${chainLocked ? " chain-locked" : ""}${special ? " special-cell" : ""}" data-cell="${index}" type="button" aria-label="Satır ${row + 1}, ${number}${type}${special ? `, ${specialOwner(type)} özel hücresi, her şekil kabul edilir` : ""}${placement ? `, ${DISK_MAP[placement.diskId].color} disk` : ", boş"}"><span class="cell-meta"><b>${number}</b><small>${type}</small></span>${specialInfo}${content}</button>`;
  }).join("");
}

function renderHands() {
  ["P1", "P2"].forEach(player => {
    const hand = state.hands[player];
    const canSelect = state.activePlayer === player;
    $(`#hand${player}`).innerHTML = hand.length
      ? hand.map(id => diskHtml(DISK_MAP[id], state.selectedDisk === id ? "selected" : "")
          .replace("type=\"button\"", `type="button" ${canSelect ? "" : "disabled"}`)).join("")
      : `<p class="empty-hand">Tüm diskler tahtada.</p>`;
    $(`#handCount${player}`).textContent = hand.length;
  });
}

function renderPlayers() {
  $("#p1Display").textContent = state.names.P1;
  $("#p2Display").textContent = state.names.P2;
  $("#targetP1").innerHTML = targetHtml(state.targets.P1, "P1");
  $("#targetP2").innerHTML = targetHtml(state.targets.P2, "P2");
  $("#aLegend").innerHTML = `<i class="legend-a"></i>A · puan hücresi`;
  $("#bLegend").innerHTML = `<i class="legend-b"></i>B · puan hücresi`;
  ["P1", "P2"].forEach(player => {
    const active = state.activePlayer === player;
    const waiting = consecutivePlacementCount(player) >= 2;
    $(`#playerPanel${player.slice(1)}`).classList.toggle("active", active);
    const selectButton = $(`#playerPanel${player.slice(1)} .select-player`);
    selectButton.textContent = waiting ? "Bekle" : active ? "Aktif" : "Seç";
    selectButton.disabled = waiting;
    selectButton.title = waiting ? "Karşı oyuncu bir disk koyduktan sonra yeniden oynayabilir." : "";
  });
  const activeNumber = state.activePlayer.slice(1);
  $("#turnText").textContent = `${state.names[state.activePlayer]} oynuyor`;
  $("#turnHint").textContent = state.selectedDisk ? "Parlayan uygun bir hücre seç" : "Elinden bir disk seç";
  const dot = $("#turnBanner .turn-dot");
  dot.className = `turn-dot p${activeNumber}`;
  $("#turnBanner").style.background = state.activePlayer === "P1" ? "var(--p1-soft)" : "var(--p2-soft)";
}

function renderStatus() {
  const filled = state.board.filter(Boolean).length;
  $("#filledCount").textContent = filled;
  $("#progressBar").style.width = `${(filled / 18) * 100}%`;
  $("#scenarioLabel").textContent = `SENARYO ${state.scenario}`;
  $("#tradeCount").textContent = `${state.trades}/6`;
  $("#tradeButton").disabled = state.trades >= 6 || !state.hands.P1.length || !state.hands.P2.length;
  $("#undoButton").disabled = state.history.length === 0;
}

function render() {
  if (!state) return;
  renderBoard();
  renderHands();
  renderPlayers();
  renderStatus();
  saveState();
}

function selectPlayer(player) {
  if (consecutivePlacementCount(player) >= 2) {
    const otherPlayer = player === "P1" ? "P2" : "P1";
    showToast(`${state.names[player]} üst üste iki disk koydu; şimdi ${state.names[otherPlayer]} oynamalı.`);
    sound("error");
    return;
  }
  state.activePlayer = player;
  state.selectedDisk = null;
  $$(".secret-card").forEach(card => card.classList.remove("revealed"));
  sound("tap");
  render();
}

function selectDisk(id) {
  if (!state.hands[state.activePlayer].includes(id)) return;
  if (consecutivePlacementCount(state.activePlayer) >= 2) {
    const otherPlayer = state.activePlayer === "P1" ? "P2" : "P1";
    showToast(`Üst üste üçüncü disk konamaz; şimdi ${state.names[otherPlayer]} oynamalı.`);
    sound("error");
    return;
  }
  state.selectedDisk = state.selectedDisk === id ? null : id;
  sound("tap");
  render();
}

function placeDisk(cellIndex) {
  if (!state.selectedDisk || state.board[cellIndex]) return;
  const disk = DISK_MAP[state.selectedDisk];
  const issue = placementIssue(disk, cellIndex, state.activePlayer);
  if (issue) {
    showToast(issue);
    sound("error");
    return;
  }
  const placedBy = state.activePlayer;
  state.history.push({
    type: "place",
    player: placedBy,
    diskId: disk.id,
    cellIndex
  });
  state.board[cellIndex] = { diskId: disk.id };
  state.hands[placedBy] = state.hands[placedBy].filter(id => id !== disk.id);
  state.selectedDisk = null;
  const filled = state.board.filter(Boolean).length;
  const mustYield = consecutivePlacementCount(placedBy) >= 2 && filled < 18;
  if (mustYield) state.activePlayer = placedBy === "P1" ? "P2" : "P1";
  sound("place");
  render();
  if (filled === 18) {
    showToast("Tahta tamamlandı — puanlar hazır!");
    setTimeout(() => openScore(true), 550);
  } else if (mustYield) {
    showToast(`${state.names[placedBy]} iki disk koydu; sıra ${state.names[state.activePlayer]}’da.`);
  }
}

function undo() {
  const action = state.history.pop();
  if (!action) return;
  if (action.type === "place") {
    state.board[action.cellIndex] = null;
    state.hands[action.player].push(action.diskId);
    state.activePlayer = action.player;
  } else if (action.type === "trade") {
    state.hands.P1 = state.hands.P1.filter(id => id !== action.p2Disk).concat(action.p1Disk);
    state.hands.P2 = state.hands.P2.filter(id => id !== action.p1Disk).concat(action.p2Disk);
    state.trades -= 1;
  }
  state.selectedDisk = null;
  sound("tap");
  showToast("Son hamle geri alındı.");
  render();
}

function openTrade() {
  if (state.trades >= 6) return showToast("6 takas sınırına ulaşıldı.");
  tradeSelection = { P1: null, P2: null };
  $("#tradeNameP1").textContent = state.names.P1;
  $("#tradeNameP2").textContent = state.names.P2;
  renderTradeHands();
  $("#tradeDialog").showModal();
}

function renderTradeHands() {
  ["P1", "P2"].forEach(player => {
    $(`#tradeHand${player}`).innerHTML = state.hands[player].map(id => diskHtml(DISK_MAP[id], tradeSelection[player] === id ? "selected" : "")).join("");
  });
  $("#confirmTrade").disabled = !(tradeSelection.P1 && tradeSelection.P2);
}

function confirmTrade() {
  const { P1: p1Disk, P2: p2Disk } = tradeSelection;
  if (!p1Disk || !p2Disk || state.trades >= 6) return;
  state.hands.P1 = state.hands.P1.filter(id => id !== p1Disk).concat(p2Disk);
  state.hands.P2 = state.hands.P2.filter(id => id !== p2Disk).concat(p1Disk);
  state.trades += 1;
  state.history.push({ type: "trade", p1Disk, p2Disk });
  $("#tradeDialog").close();
  sound("place");
  showToast(`Takas kabul edildi · ${state.trades}/6`);
  render();
}

function calculateScore(player) {
  const target = TARGETS[state.targets[player]];
  return state.board.reduce((total, placement, index) => {
    if (!placement) return total;
    const disk = DISK_MAP[placement.diskId];
    const row = Math.floor(index / 6);
    const type = BOARD_TYPES[row][index % 6];
    if (type !== target.cell) return total;
    if (disk.shape === target.primary) return total + target.primaryPoints;
    if (disk.shape === target.secondary) return total + target.secondaryPoints;
    return total;
  }, 0);
}

function openScore(forceReveal = false) {
  const complete = state.board.every(Boolean);
  const p1Target = calculateScore("P1");
  const p2Target = calculateScore("P2");
  $("#scoreTitle").textContent = complete ? "Üç zincir tamamlandı!" : "Puanlar hâlâ gizli";
  $("#scoreNameP1").textContent = state.names.P1;
  $("#scoreNameP2").textContent = state.names.P2;
  $("#scoreP1").textContent = complete ? p1Target : "?";
  $("#scoreP2").textContent = complete ? p2Target : "?";
  $("#scoreDetailP1").textContent = complete ? `Hedef puanı: ${p1Target}` : "Oyun sonunda açıklanır";
  $("#scoreDetailP2").textContent = complete ? `Hedef puanı: ${p2Target}` : "Oyun sonunda açıklanır";
  $("#scoreDialog").showModal();
  if (forceReveal) sound("finish");
}

function targetsForScenario(scenario) {
  return scenario === 1
    ? { P1: "circles", P2: "polygons" }
    : { P1: "polygons", P2: "circles" };
}

function physicalSpecialOwner(type) {
  const targets = targetsForScenario(physicalScoreState.scenario);
  return Object.keys(targets).find(player => TARGETS[targets[player]].cell === type);
}

function physicalTargetScore(player) {
  const targetKey = targetsForScenario(physicalScoreState.scenario)[player];
  const target = TARGETS[targetKey];
  return physicalScoreState.shapes.reduce((total, shape, index) => {
    if (!shape) return total;
    const row = Math.floor(index / 6);
    if (BOARD_TYPES[row][index % 6] !== target.cell) return total;
    if (shape === target.primary) return total + target.primaryPoints;
    if (shape === target.secondary) return total + target.secondaryPoints;
    return total;
  }, 0);
}

function renderPhysicalBoard() {
  $("#physicalBoard").innerHTML = physicalScoreState.shapes.map((shape, index) => {
    const row = Math.floor(index / 6);
    const type = BOARD_TYPES[row][index % 6];
    const number = cellNumber(index);
    const scoreable = type !== "N";
    const special = SPECIAL_CELLS[index];
    const owner = special ? physicalSpecialOwner(type) : null;
    const confidence = physicalScoreState.detectionConfidence[index];
    const detectionClass = confidence === null ? "" : confidence < .34 ? " uncertain" : " detected";
    const specialSeal = special ? `<span class="physical-special-seal ${owner.toLowerCase()}">Ö${owner}</span>` : "";
    const shapeContent = shape
      ? `<span class="physical-cell-shape"><strong>${SHAPE_ICONS[shape]}</strong><small>${shape}</small></span>`
      : scoreable ? `<span class="physical-cell-shape physical-cell-empty">＋</span>` : "";
    return `<button class="physical-cell type-${type}${shape ? " assigned" : ""}${detectionClass}${physicalScoreState.selectedCell === index ? " selected" : ""}" data-physical-cell="${index}" type="button" ${scoreable ? "" : "disabled"} aria-label="${number}${type}${shape ? `, ${shape}` : scoreable ? ", şekil seçilmedi" : ", nötr"}"><span class="physical-cell-meta"><b>${number}</b><small>${type}</small></span>${specialSeal}${shapeContent}</button>`;
  }).join("");
}

function renderPhysicalShapePicker() {
  const picker = $("#physicalShapePicker");
  const index = physicalScoreState.selectedCell;
  if (index === null) {
    picker.innerHTML = "<span>Önce bir A/B hücresi seçin.</span>";
    return;
  }
  const row = Math.floor(index / 6);
  const label = `${cellNumber(index)}${BOARD_TYPES[row][index % 6]}`;
  picker.innerHTML = `<span><b>${label}</b></span>${Object.keys(SHAPE_ICONS).map(shape => `<button class="shape-option" data-physical-shape="${shape}" type="button" aria-label="${label} için ${shape} seç"><b>${SHAPE_ICONS[shape]}</b><small>${shape}</small></button>`).join("")}<button class="shape-option clear" data-physical-shape="" type="button">Temizle</button>`;
}

function renderPhysicalResult() {
  const scoreableCount = BOARD_TYPES.flat().filter(type => type !== "N").length;
  const assignedCount = physicalScoreState.shapes.filter((shape, index) => {
    const row = Math.floor(index / 6);
    return shape && BOARD_TYPES[row][index % 6] !== "N";
  }).length;
  const p1Target = physicalTargetScore("P1");
  const p2Target = physicalTargetScore("P2");
  $("#physicalProgress").textContent = `${assignedCount} / ${scoreableCount} işlendi`;
  $("#physicalNameP1").textContent = physicalScoreState.names.P1;
  $("#physicalNameP2").textContent = physicalScoreState.names.P2;
  $("#physicalScoreP1").textContent = p1Target;
  $("#physicalScoreP2").textContent = p2Target;
  $("#physicalDetailP1").textContent = `Hedef puanı: ${p1Target}`;
  $("#physicalDetailP2").textContent = `Hedef puanı: ${p2Target}`;
}

function renderPhysicalScore() {
  $("#physicalScenario").value = String(physicalScoreState.scenario);
  $("#physicalP1Name").value = physicalScoreState.names.P1;
  $("#physicalP2Name").value = physicalScoreState.names.P2;
  renderPhysicalBoard();
  renderPhysicalShapePicker();
  renderPhysicalResult();
}

function openPhysicalScore() {
  if (!physicalScoreState.initialized) {
    physicalScoreState.scenario = state?.scenario || setupScenario;
    physicalScoreState.names = state
      ? { ...state.names }
      : {
          P1: $("#p1Name").value.trim() || "Oyuncu 1",
          P2: $("#p2Name").value.trim() || "Oyuncu 2"
        };
    physicalScoreState.initialized = true;
  }
  renderPhysicalScore();
  const dialog = $("#physicalScoreDialog");
  dialog.showModal();
  dialog.scrollTop = 0;
}

function selectPhysicalShape(shape) {
  const index = physicalScoreState.selectedCell;
  if (index === null) return;
  physicalScoreState.shapes[index] = shape || null;
  physicalScoreState.detectionConfidence[index] = shape ? 1 : null;
  if (shape) {
    const scoreableCells = physicalScoreState.shapes
      .map((_, cellIndex) => cellIndex)
      .filter(cellIndex => BOARD_TYPES[Math.floor(cellIndex / 6)][cellIndex % 6] !== "N");
    const currentPosition = scoreableCells.indexOf(index);
    const orderedNext = scoreableCells.slice(currentPosition + 1).concat(scoreableCells.slice(0, currentPosition));
    physicalScoreState.selectedCell = orderedNext.find(cellIndex => !physicalScoreState.shapes[cellIndex]) ?? index;
  }
  renderPhysicalScore();
}

function setPhysicalAnalysisStatus(kind, title, detail) {
  const status = $("#physicalAnalysisStatus");
  status.className = `photo-analysis-status ${kind}`;
  status.innerHTML = `<b>${escapeHtml(title)}</b><span>${escapeHtml(detail)}</span>`;
}

function drawPhysicalPhotoCanvas(showDetections = true) {
  const image = $("#physicalPhoto");
  const canvas = $("#physicalPhotoCanvas");
  if (!image.naturalWidth) return;
  const maxSide = 600;
  const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
  const sourceWidth = Math.round(image.naturalWidth * scale);
  const sourceHeight = Math.round(image.naturalHeight * scale);
  const sideways = physicalScoreState.rotation % 180 !== 0;
  canvas.width = sideways ? sourceHeight : sourceWidth;
  canvas.height = sideways ? sourceWidth : sourceHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.save();
  if (physicalScoreState.rotation === 90) {
    context.translate(canvas.width, 0);
    context.rotate(Math.PI / 2);
  } else if (physicalScoreState.rotation === 180) {
    context.translate(canvas.width, canvas.height);
    context.rotate(Math.PI);
  } else if (physicalScoreState.rotation === 270) {
    context.translate(0, canvas.height);
    context.rotate(-Math.PI / 2);
  }
  context.drawImage(image, 0, 0, sourceWidth, sourceHeight);
  context.restore();

  if (!showDetections || !physicalScoreState.detections.length) return;
  context.save();
  context.font = "700 12px DM Sans, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  physicalScoreState.detections.forEach(detection => {
    const confidence = physicalScoreState.detectionConfidence[detection.index] ?? 0;
    const color = confidence < .34 ? "#745ca7" : "#43a04d";
    context.beginPath();
    context.arc(detection.x, detection.y, detection.radius, 0, Math.PI * 2);
    context.strokeStyle = color;
    context.lineWidth = Math.max(3, canvas.width / 300);
    context.stroke();
    const label = `${detection.number} · ${SHAPE_ICONS[detection.shape] || "?"}`;
    const width = context.measureText(label).width + 14;
    const labelY = Math.max(13, detection.y - detection.radius - 11);
    context.fillStyle = "rgba(19,42,56,.88)";
    context.fillRect(detection.x - width / 2, labelY - 10, width, 20);
    context.fillStyle = "white";
    context.fillText(label, detection.x, labelY);
  });
  context.restore();
}

function rgbToHsv(red, green, blue) {
  const r = red / 255;
  const g = green / 255;
  const b = blue / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let hue = 0;
  if (delta) {
    if (max === r) hue = 60 * (((g - b) / delta) % 6);
    else if (max === g) hue = 60 * ((b - r) / delta + 2);
    else hue = 60 * ((r - g) / delta + 4);
  }
  if (hue < 0) hue += 360;
  return { h: hue / 2, s: max ? (delta / max) * 255 : 0, v: max * 255 };
}

const DETECTABLE_COLORS = ["Kırmızı", "Turuncu", "Sarı", "Yeşil", "Mavi", "Lacivert"];

function detectedColorForPixel(red, green, blue) {
  const { h, s, v } = rgbToHsv(red, green, blue);
  if ((h < 8 || h > 174) && s > 130 && v > 80) return 0;
  if (h >= 8 && h < 18 && s > 160 && v > 120) return 1;
  if (h >= 18 && h < 32 && s > 150 && v > 120) return 2;
  if (h >= 42 && h < 80 && s > 90 && v > 70) return 3;
  if (h >= 90 && h < 112 && s > 110 && v > 105) return 4;
  if (h >= 108 && h < 145 && s > 65 && v < 155) return 5;
  return -1;
}

function findColoredDiskComponents(context) {
  const { width, height } = context.canvas;
  const image = context.getImageData(0, 0, width, height);
  const classes = new Int8Array(width * height);
  classes.fill(-1);
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    classes[index] = detectedColorForPixel(image.data[offset], image.data[offset + 1], image.data[offset + 2]);
  }
  const visited = new Uint8Array(classes.length);
  const stack = new Int32Array(classes.length);
  const components = [];
  const minimumArea = width * height * .0024;
  const maximumArea = width * height * .04;
  for (let start = 0; start < classes.length; start += 1) {
    const colorIndex = classes[start];
    if (colorIndex < 0 || visited[start]) continue;
    let stackSize = 1;
    let area = 0;
    let sumX = 0;
    let sumY = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    stack[0] = start;
    visited[start] = 1;
    while (stackSize) {
      const current = stack[--stackSize];
      const x = current % width;
      const y = Math.floor(current / width);
      area += 1;
      sumX += x;
      sumY += y;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      const neighbors = [current - 1, current + 1, current - width, current + width];
      neighbors.forEach((neighbor, direction) => {
        if (neighbor < 0 || neighbor >= classes.length || visited[neighbor] || classes[neighbor] !== colorIndex) return;
        if (direction === 0 && x === 0) return;
        if (direction === 1 && x === width - 1) return;
        visited[neighbor] = 1;
        stack[stackSize++] = neighbor;
      });
    }
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    if (area < minimumArea || area > maximumArea || boxWidth < width * .035 || boxWidth > width * .19 || boxHeight < height * .035 || boxHeight > height * .21) continue;
    components.push({ color: DETECTABLE_COLORS[colorIndex], x: sumX / area, y: sumY / area, area, boxWidth, boxHeight });
  }
  const selected = DETECTABLE_COLORS.flatMap(color => components.filter(component => component.color === color).sort((a, b) => b.area - a.area).slice(0, 2));
  const counts = Object.fromEntries(DETECTABLE_COLORS.map(color => [color, selected.filter(component => component.color === color).length]));
  const found = Object.values(counts).reduce((sum, count) => sum + count, 0);
  if (found < 3) throw new Error(`12 renkli diskten yalnızca ${found} tanesi bulundu. Işığı artırıp tahtayı biraz daha tepeden çekin.`);
  return { components: selected, image, found };
}

function columnCombinations(size, choose, start = 0, prefix = []) {
  if (prefix.length === choose) return [prefix];
  const combinations = [];
  for (let value = start; value <= size - (choose - prefix.length); value += 1) {
    combinations.push(...columnCombinations(size, choose, value + 1, [...prefix, value]));
  }
  return combinations;
}

function linearFit(points, coordinate) {
  const count = points.length;
  const meanColumn = points.reduce((sum, point) => sum + point.column, 0) / count;
  const meanValue = points.reduce((sum, point) => sum + point[coordinate], 0) / count;
  const denominator = points.reduce((sum, point) => sum + (point.column - meanColumn) ** 2, 0) || 1;
  const slope = points.reduce((sum, point) => sum + (point.column - meanColumn) * (point[coordinate] - meanValue), 0) / denominator;
  const intercept = meanValue - slope * meanColumn;
  const error = points.reduce((sum, point) => sum + (point[coordinate] - (intercept + slope * point.column)) ** 2, 0) / count;
  return { intercept, slope, error };
}

function mapColoredDisksToGrid(components) {
  let centers = [Math.min(...components.map(disk => disk.y)), components.map(disk => disk.y).sort((a, b) => a - b)[Math.floor(components.length / 2)], Math.max(...components.map(disk => disk.y))];
  let rows = [];
  for (let iteration = 0; iteration < 12; iteration += 1) {
    rows = [[], [], []];
    components.forEach(disk => {
      const distances = centers.map(center => Math.abs(disk.y - center));
      const rowIndex = distances.indexOf(Math.min(...distances));
      rows[rowIndex].push(disk);
    });
    if (rows.some(row => !row.length)) throw new Error("Renkli diskler 3 tahta satırına ayrılamadı. Fotoğrafı döndürüp yeniden deneyin.");
    centers = rows.map(row => row.reduce((sum, disk) => sum + disk.y, 0) / row.length);
  }
  rows = rows.map(row => row.sort((a, b) => a.x - b.x));
  const rowCenters = rows.map(row => row.reduce((sum, disk) => sum + disk.y, 0) / row.length);
  if (rowCenters[1] - rowCenters[0] < 35 || rowCenters[2] - rowCenters[1] < 35) throw new Error("Renkli diskler 3 tahta satırına ayrılamadı. Fotoğrafı döndürüp yeniden deneyin.");
  const rowOptions = rows.map((row, rowIndex) => {
    const options = columnCombinations(6, row.length).filter(columns => columns.every((column, index) => {
      const number = BOARD_NUMBERS[rowIndex][column];
      return DISKS.some(disk => disk.color === row[index].color && disk.number === number);
    })).map(columns => {
      const points = row.map((disk, index) => ({ ...disk, column: columns[index] }));
      return { points };
    });
    if (!options.length) throw new Error(`Satır ${rowIndex + 1} renk dizilimi sabit disk setiyle eşleşmedi.`);
    return options;
  });

  const combinations = rowOptions[0].flatMap(first => rowOptions[1].flatMap(second => rowOptions[2].map(third => [first, second, third])));
  const ranked = combinations.map(choice => {
    const colorNumbers = Object.fromEntries(DETECTABLE_COLORS.map(color => [color, []]));
    choice.forEach((option, rowIndex) => option.points.forEach(point => colorNumbers[point.color].push(BOARD_NUMBERS[rowIndex][point.column])));
    if (Object.values(colorNumbers).some(numbers => new Set(numbers).size !== numbers.length)) return null;
    const directFits = choice.map(option => option.points.length > 1 ? linearFit(option.points, "x") : null);
    const slopes = directFits.filter(Boolean).map(fit => fit.slope).filter(slope => slope > 0).sort((a, b) => a - b);
    if (!slopes.length) return null;
    const commonSlope = slopes[Math.floor(slopes.length / 2)];
    const completed = choice.map((option, rowIndex) => {
      const xFit = directFits[rowIndex] || { slope: commonSlope, intercept: option.points[0].x - commonSlope * option.points[0].column, error: 0 };
      const yFit = option.points.length > 1 ? linearFit(option.points, "y") : { slope: 0, intercept: option.points[0].y, error: 0 };
      return { ...option, xFit, yFit };
    });
    const leftFit = linearFit(completed.map((option, rowIndex) => ({ column: rowIndex, x: option.xFit.intercept })), "x");
    const rightFit = linearFit(completed.map((option, rowIndex) => ({ column: rowIndex, x: option.xFit.intercept + option.xFit.slope * 5 })), "x");
    const spacingPenalty = completed.reduce((sum, option) => sum + ((option.xFit.slope - commonSlope) / commonSlope) ** 2 * 120, 0);
    const error = completed.reduce((sum, option) => sum + option.xFit.error + option.yFit.error, 0) + leftFit.error + rightFit.error + spacingPenalty;
    return { completed, error };
  }).filter(Boolean).sort((a, b) => a.error - b.error);
  if (!ranked.length) throw new Error("Renkli disklerin hücreleri sabit disk setiyle eşleştirilemedi.");
  return ranked[0].completed;
}

function sampleCellBrightness(image, center, radius) {
  const values = [];
  for (let y = Math.max(0, Math.floor(center.y - radius * .35)); y <= Math.min(image.height - 1, Math.ceil(center.y + radius * .35)); y += 2) {
    for (let x = Math.max(0, Math.floor(center.x - radius * .88)); x <= Math.min(image.width - 1, Math.ceil(center.x + radius * .88)); x += 2) {
      const normalizedX = (x - center.x) / radius;
      if (Math.abs(normalizedX) < .5 || Math.abs(normalizedX) > .88) continue;
      const offset = (y * image.width + x) * 4;
      const red = image.data[offset];
      const green = image.data[offset + 1];
      const blue = image.data[offset + 2];
      values.push(red * .299 + green * .587 + blue * .114);
    }
  }
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)] || 0;
}

function detectPhysicalBoard() {
  const canvas = $("#physicalPhotoCanvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const { components, image, found } = findColoredDiskComponents(context);
  const fittedRows = mapColoredDisksToGrid(components);
  const centers = fittedRows.flatMap((row, rowIndex) => Array.from({ length: 6 }, (_, column) => ({
    index: rowIndex * 6 + column,
    row: rowIndex,
    column,
    number: BOARD_NUMBERS[rowIndex][column],
    x: row.xFit.intercept + row.xFit.slope * column,
    y: row.yFit.intercept + row.yFit.slope * column,
    radius: Math.abs(row.xFit.slope) * .42
  })));
  const assignments = Array(18).fill(null);
  const confidences = Array(18).fill(null);
  fittedRows.forEach((row, rowIndex) => row.points.forEach(point => {
    const index = rowIndex * 6 + point.column;
    assignments[index] = DISKS.find(disk => disk.color === point.color && disk.number === cellNumber(index));
    confidences[index] = .95;
  }));

  for (let number = 1; number <= 6; number += 1) {
    const numberCells = centers.filter(center => center.number === number && !assignments[center.index]);
    const usedColors = new Set(assignments.filter(disk => disk?.number === number).map(disk => disk.color));
    const remaining = DISKS.filter(disk => disk.number === number && !usedColors.has(disk.color));
    if (numberCells.length !== remaining.length) throw new Error(`${number} numaralı diskler tekil olarak eşleştirilemedi.`);
    if (numberCells.length === 1) {
      assignments[numberCells[0].index] = remaining[0];
      confidences[numberCells[0].index] = .68;
      continue;
    }
    const neutralColors = new Set(["Siyah", "Gri", "Beyaz"]);
    if (!remaining.every(disk => neutralColors.has(disk.color))) continue;
    const cellsByBrightness = numberCells.map(center => ({ center, brightness: sampleCellBrightness(image, center, center.radius) })).sort((a, b) => a.brightness - b.brightness);
    const disksByBrightness = [...remaining].sort((a, b) => ({ Siyah: 0, Gri: 1, Beyaz: 2 }[a.color] ?? 3) - ({ Siyah: 0, Gri: 1, Beyaz: 2 }[b.color] ?? 3));
    cellsByBrightness.forEach((item, index) => {
      assignments[item.center.index] = disksByBrightness[index];
      confidences[item.center.index] = Math.abs(cellsByBrightness[0].brightness - cellsByBrightness.at(-1).brightness) > 18 ? .72 : .28;
    });
  }
  return {
    shapes: assignments.map(disk => disk?.shape || null),
    confidences,
    detectedColorCount: found,
    assignedCount: assignments.filter(Boolean).length,
    detections: centers.filter(center => assignments[center.index]).map(center => ({ ...center, color: assignments[center.index].color, shape: assignments[center.index].shape }))
  };
}

function analyzePhysicalPhoto() {
  if (!physicalScoreState.photoUrl || physicalScoreState.analyzing) return;
  physicalScoreState.analyzing = true;
  physicalScoreState.detections = [];
  drawPhysicalPhotoCanvas(false);
  $("#photoAnalysisOverlay").classList.remove("hidden");
  setPhysicalAnalysisStatus("working", "Analiz ediliyor", "18 disk ve renkleri cihazınızda aranıyor…");
  try {
    const result = detectPhysicalBoard();
    physicalScoreState.shapes = result.shapes;
    physicalScoreState.detectionConfidence = result.confidences;
    physicalScoreState.detections = result.detections;
    physicalScoreState.selectedCell = null;
    const uncertain = result.confidences.filter(confidence => confidence !== null && confidence < .34).length;
    renderPhysicalScore();
    drawPhysicalPhotoCanvas(true);
    const partial = result.assignedCount < 18;
    if (partial) {
      setPhysicalAnalysisStatus(
        "warning",
        `${result.assignedCount} / 18 disk yerleştirildi`,
        `${result.detectedColorCount}/12 renkli disk net bulundu. Eksik A/B hücrelerini sağdaki tahtadan tamamlayın.`
      );
      showToast("Bulunan diskler yerleştirildi; eksikleri elle tamamlayabilirsiniz.");
    } else {
      setPhysicalAnalysisStatus(
        uncertain ? "warning" : "success",
        "18 / 18 disk okundu",
        uncertain ? `${uncertain} okuma mor renkle işaretlendi; puanı onaylamadan önce kontrol edin.` : "Tüm şekiller bulundu ve puanlar otomatik hesaplandı."
      );
      showToast("Fotoğraf otomatik okundu, puanlar hazır.");
    }
  } catch (error) {
    physicalScoreState.detections = [];
    physicalScoreState.detectionConfidence = Array(18).fill(null);
    drawPhysicalPhotoCanvas(false);
    setPhysicalAnalysisStatus("warning", "Otomatik okuma tamamlanamadı", error.message || "Fotoğrafı düzeltip yeniden deneyin.");
    showToast("Tahta net okunamadı; fotoğrafı döndürüp yeniden tarayın.");
  } finally {
    physicalScoreState.analyzing = false;
    $("#photoAnalysisOverlay").classList.add("hidden");
  }
}

function clearPhysicalPhoto() {
  if (physicalScoreState.photoUrl) URL.revokeObjectURL(physicalScoreState.photoUrl);
  physicalScoreState.photoUrl = null;
  physicalScoreState.rotation = 0;
  physicalScoreState.detections = [];
  physicalScoreState.detectionConfidence = Array(18).fill(null);
  $("#physicalPhotoInput").value = "";
  $("#physicalPhoto").removeAttribute("src");
  const canvas = $("#physicalPhotoCanvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  $("#photoPreview").classList.add("hidden");
  $("#photoDropzone").classList.remove("hidden");
  setPhysicalAnalysisStatus("idle", "Hazır", "Tahtayı yukarıdan, 18 diskin tamamı görünecek şekilde çekin.");
}

async function loadPhysicalPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return showToast("Lütfen bir görsel dosyası seçin.");
  if (file.size > 20 * 1024 * 1024) return showToast("Fotoğraf 20 MB’den küçük olmalı.");
  if (physicalScoreState.photoUrl) URL.revokeObjectURL(physicalScoreState.photoUrl);
  physicalScoreState.photoUrl = URL.createObjectURL(file);
  physicalScoreState.rotation = 0;
  physicalScoreState.detections = [];
  physicalScoreState.detectionConfidence = Array(18).fill(null);
  const image = $("#physicalPhoto");
  image.src = physicalScoreState.photoUrl;
  $("#photoDropzone").classList.add("hidden");
  $("#photoPreview").classList.remove("hidden");
  try {
    await image.decode();
  } catch (_) {
    await new Promise(resolve => image.addEventListener("load", resolve, { once: true }));
  }
  drawPhysicalPhotoCanvas(false);
  analyzePhysicalPhoto();
}

function rotatePhysicalPhoto(delta) {
  physicalScoreState.rotation = (physicalScoreState.rotation + delta + 360) % 360;
  physicalScoreState.detections = [];
  physicalScoreState.detectionConfidence = Array(18).fill(null);
  drawPhysicalPhotoCanvas(false);
  analyzePhysicalPhoto();
}

function resetPhysicalScore() {
  physicalScoreState.shapes = Array(18).fill(null);
  physicalScoreState.detectionConfidence = Array(18).fill(null);
  physicalScoreState.detections = [];
  physicalScoreState.selectedCell = null;
  clearPhysicalPhoto();
  renderPhysicalScore();
  showToast("Fiziksel puan formu temizlendi.");
}

async function copyPhysicalResult() {
  const p1Target = physicalTargetScore("P1");
  const p2Target = physicalTargetScore("P2");
  const assigned = physicalScoreState.shapes.filter((shape, index) => shape && BOARD_TYPES[Math.floor(index / 6)][index % 6] !== "N").length;
  const text = `USTA fiziksel oturum · Senaryo ${physicalScoreState.scenario}\n${physicalScoreState.names.P1}: ${p1Target}\n${physicalScoreState.names.P2}: ${p2Target}\nİşlenen puan hücresi: ${assigned}/12`;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  showToast("Puan özeti kopyalandı.");
}

function startGame(useSaved = false) {
  if (useSaved) {
    state = loadSavedState();
    if (!state) return;
  } else {
    state = initialState(setupScenario, {
      P1: $("#p1Name").value.trim(),
      P2: $("#p2Name").value.trim()
    });
  }
  if (consecutivePlacementCount(state.activePlayer) >= 2) {
    state.activePlayer = state.activePlayer === "P1" ? "P2" : "P1";
  }
  $("#setupScreen").classList.add("hidden");
  $("#gameScreen").classList.remove("hidden");
  $("#soundButton").setAttribute("aria-pressed", String(state.sound));
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function newGame() {
  if (!confirm("Mevcut oyun silinsin ve başlangıç ekranına dönülsün mü?")) return;
  localStorage.removeItem(STORAGE_KEY);
  state = null;
  $("#scoreDialog").close();
  $("#gameScreen").classList.add("hidden");
  $("#setupScreen").classList.remove("hidden");
  $("#resumeButton").classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2400);
}

function sound(type) {
  if (!state?.sound) return;
  try {
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    const settings = {
      tap: [310, .035], place: [480, .07], error: [140, .08], finish: [660, .22]
    }[type] || [300, .04];
    oscillator.frequency.value = settings[0];
    oscillator.type = type === "finish" ? "sine" : "triangle";
    gain.gain.setValueAtTime(.045, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + settings[1]);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + settings[1]);
  } catch (_) {}
}

$$('[data-scenario]').forEach(button => button.addEventListener("click", () => {
  setupScenario = Number(button.dataset.scenario);
  $$('[data-scenario]').forEach(card => {
    const selected = card === button;
    card.classList.toggle("selected", selected);
    card.setAttribute("aria-checked", String(selected));
  });
}));

$("#startButton").addEventListener("click", () => startGame(false));
$("#resumeButton").addEventListener("click", () => startGame(true));
$("#physicalScoreButton").addEventListener("click", openPhysicalScore);
$("#rulesButton").addEventListener("click", () => $("#rulesDialog").showModal());
$("#tradeButton").addEventListener("click", openTrade);
$("#confirmTrade").addEventListener("click", confirmTrade);
$("#undoButton").addEventListener("click", undo);
$("#scoreButton").addEventListener("click", () => openScore(false));
$("#newGameButton").addEventListener("click", newGame);

$("#physicalScenario").addEventListener("change", event => {
  physicalScoreState.scenario = Number(event.target.value);
  renderPhysicalScore();
});

$("#physicalP1Name").addEventListener("input", event => {
  physicalScoreState.names.P1 = event.target.value.trim() || "Oyuncu 1";
  renderPhysicalResult();
});

$("#physicalP2Name").addEventListener("input", event => {
  physicalScoreState.names.P2 = event.target.value.trim() || "Oyuncu 2";
  renderPhysicalResult();
});

$("#physicalPhotoInput").addEventListener("change", event => loadPhysicalPhoto(event.target.files[0]));
$("#rotatePhotoLeft").addEventListener("click", () => rotatePhysicalPhoto(-90));
$("#rotatePhotoRight").addEventListener("click", () => rotatePhysicalPhoto(90));
$("#analyzePhysicalPhoto").addEventListener("click", analyzePhysicalPhoto);
$("#removePhoto").addEventListener("click", clearPhysicalPhoto);
$("#resetPhysicalScore").addEventListener("click", resetPhysicalScore);
$("#copyPhysicalResult").addEventListener("click", copyPhysicalResult);

$("#physicalBoard").addEventListener("click", event => {
  const cell = event.target.closest("[data-physical-cell]");
  if (!cell || cell.disabled) return;
  physicalScoreState.selectedCell = Number(cell.dataset.physicalCell);
  renderPhysicalScore();
});

$("#physicalShapePicker").addEventListener("click", event => {
  const option = event.target.closest("[data-physical-shape]");
  if (!option) return;
  selectPhysicalShape(option.dataset.physicalShape);
});

const photoDropzone = $("#photoDropzone");
["dragenter", "dragover"].forEach(eventName => photoDropzone.addEventListener(eventName, event => {
  event.preventDefault();
  photoDropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(eventName => photoDropzone.addEventListener(eventName, event => {
  event.preventDefault();
  photoDropzone.classList.remove("dragover");
}));
photoDropzone.addEventListener("drop", event => loadPhysicalPhoto(event.dataTransfer.files[0]));

$("#soundButton").addEventListener("click", event => {
  if (!state) return;
  state.sound = !state.sound;
  event.currentTarget.setAttribute("aria-pressed", String(state.sound));
  event.currentTarget.setAttribute("aria-label", state.sound ? "Sesi kapat" : "Sesi aç");
  if (state.sound) sound("tap");
  saveState();
});

document.addEventListener("click", event => {
  const closeButton = event.target.closest("[data-close]");
  if (closeButton) closeButton.closest("dialog").close();

  const playerButton = event.target.closest(".select-player");
  if (playerButton) selectPlayer(playerButton.dataset.player);

  const secretCard = event.target.closest(".secret-card");
  if (secretCard) {
    const wasRevealed = secretCard.classList.contains("revealed");
    $$(".secret-card").forEach(card => card.classList.remove("revealed"));
    secretCard.classList.toggle("revealed", !wasRevealed);
    sound("tap");
  }

  const handDisk = event.target.closest(".hand .disk-button");
  if (handDisk) selectDisk(handDisk.dataset.disk);

  const boardCell = event.target.closest(".board-cell");
  if (boardCell) placeDisk(Number(boardCell.dataset.cell));

  const tradeDisk = event.target.closest(".trade-hand .disk-button");
  if (tradeDisk) {
    const player = tradeDisk.closest(".trade-hand").id.endsWith("P1") ? "P1" : "P2";
    tradeSelection[player] = tradeSelection[player] === tradeDisk.dataset.disk ? null : tradeDisk.dataset.disk;
    renderTradeHands();
    sound("tap");
  }
});

document.addEventListener("keydown", event => {
  if (!state || $("#gameScreen").classList.contains("hidden")) return;
  if (event.key === "1") selectPlayer("P1");
  if (event.key === "2") selectPlayer("P2");
  if (event.key.toLowerCase() === "t" && !$("#tradeButton").disabled) openTrade();
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") undo();
});

if (loadSavedState()) $("#resumeButton").classList.remove("hidden");
