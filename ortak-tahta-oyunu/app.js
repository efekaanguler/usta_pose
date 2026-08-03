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
  selectedCell: null,
  complete: true,
  photoUrl: null,
  rotation: 0
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
  $("#aLegend").innerHTML = `<i class="legend-a"></i>A · bonus hücresi`;
  $("#bLegend").innerHTML = `<i class="legend-b"></i>B · bonus hücresi`;
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
    showToast("Tahta tamamlandı — iki oyuncuya da +20!");
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
  const bonus = complete ? 20 : 0;
  $("#scoreTitle").textContent = complete ? "Üç zincir tamamlandı!" : "Puanlar hâlâ gizli";
  $("#scoreNameP1").textContent = state.names.P1;
  $("#scoreNameP2").textContent = state.names.P2;
  $("#scoreP1").textContent = complete ? p1Target + bonus : "?";
  $("#scoreP2").textContent = complete ? p2Target + bonus : "?";
  $("#scoreDetailP1").textContent = complete ? `Hedef ${p1Target} + ortak 20` : "Oyun sonunda açıklanır";
  $("#scoreDetailP2").textContent = complete ? `Hedef ${p2Target} + ortak 20` : "Oyun sonunda açıklanır";
  $("#completionBonus").classList.toggle("earned", complete);
  $("#completionBonus small").textContent = complete ? "Kazanıldı — iki skora da eklendi." : "Üç zincir tamamlandığında iki oyuncuya da eklenir.";
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
    const specialSeal = special ? `<span class="physical-special-seal ${owner.toLowerCase()}">Ö${owner}</span>` : "";
    const shapeContent = shape
      ? `<span class="physical-cell-shape"><strong>${SHAPE_ICONS[shape]}</strong><small>${shape}</small></span>`
      : scoreable ? `<span class="physical-cell-shape physical-cell-empty">＋</span>` : "";
    return `<button class="physical-cell type-${type}${shape ? " assigned" : ""}${physicalScoreState.selectedCell === index ? " selected" : ""}" data-physical-cell="${index}" type="button" ${scoreable ? "" : "disabled"} aria-label="${number}${type}${shape ? `, ${shape}` : scoreable ? ", şekil seçilmedi" : ", nötr"}"><span class="physical-cell-meta"><b>${number}</b><small>${type}</small></span>${specialSeal}${shapeContent}</button>`;
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
  const bonus = physicalScoreState.complete ? 20 : 0;
  $("#physicalProgress").textContent = `${assignedCount} / ${scoreableCount} işlendi`;
  $("#physicalNameP1").textContent = physicalScoreState.names.P1;
  $("#physicalNameP2").textContent = physicalScoreState.names.P2;
  $("#physicalScoreP1").textContent = p1Target + bonus;
  $("#physicalScoreP2").textContent = p2Target + bonus;
  $("#physicalDetailP1").textContent = `Hedef ${p1Target}${bonus ? " + ortak 20" : ""}`;
  $("#physicalDetailP2").textContent = `Hedef ${p2Target}${bonus ? " + ortak 20" : ""}`;
}

function renderPhysicalScore() {
  $("#physicalScenario").value = String(physicalScoreState.scenario);
  $("#physicalP1Name").value = physicalScoreState.names.P1;
  $("#physicalP2Name").value = physicalScoreState.names.P2;
  $("#physicalComplete").checked = physicalScoreState.complete;
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

function clearPhysicalPhoto() {
  if (physicalScoreState.photoUrl) URL.revokeObjectURL(physicalScoreState.photoUrl);
  physicalScoreState.photoUrl = null;
  physicalScoreState.rotation = 0;
  $("#physicalPhotoInput").value = "";
  $("#physicalPhoto").removeAttribute("src");
  $("#photoPreview").classList.add("hidden");
  $("#photoDropzone").classList.remove("hidden");
}

function loadPhysicalPhoto(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return showToast("Lütfen bir görsel dosyası seçin.");
  if (file.size > 20 * 1024 * 1024) return showToast("Fotoğraf 20 MB’den küçük olmalı.");
  if (physicalScoreState.photoUrl) URL.revokeObjectURL(physicalScoreState.photoUrl);
  physicalScoreState.photoUrl = URL.createObjectURL(file);
  physicalScoreState.rotation = 0;
  const image = $("#physicalPhoto");
  image.src = physicalScoreState.photoUrl;
  image.style.transform = "rotate(0deg)";
  $("#photoDropzone").classList.add("hidden");
  $("#photoPreview").classList.remove("hidden");
}

function rotatePhysicalPhoto(delta) {
  physicalScoreState.rotation = (physicalScoreState.rotation + delta + 360) % 360;
  $("#physicalPhoto").style.transform = `rotate(${physicalScoreState.rotation}deg)`;
}

function resetPhysicalScore() {
  physicalScoreState.shapes = Array(18).fill(null);
  physicalScoreState.selectedCell = null;
  physicalScoreState.complete = true;
  clearPhysicalPhoto();
  renderPhysicalScore();
  showToast("Fiziksel puan formu temizlendi.");
}

async function copyPhysicalResult() {
  const p1Target = physicalTargetScore("P1");
  const p2Target = physicalTargetScore("P2");
  const bonus = physicalScoreState.complete ? 20 : 0;
  const assigned = physicalScoreState.shapes.filter(Boolean).length;
  const text = `USTA fiziksel oturum · Senaryo ${physicalScoreState.scenario}\n${physicalScoreState.names.P1}: ${p1Target + bonus} (hedef ${p1Target}${bonus ? " + ortak 20" : ""})\n${physicalScoreState.names.P2}: ${p2Target + bonus} (hedef ${p2Target}${bonus ? " + ortak 20" : ""})\nİşlenen bonus hücresi: ${assigned}/12`;
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

$("#physicalComplete").addEventListener("change", event => {
  physicalScoreState.complete = event.target.checked;
  renderPhysicalResult();
});

$("#physicalPhotoInput").addEventListener("change", event => loadPhysicalPhoto(event.target.files[0]));
$("#rotatePhotoLeft").addEventListener("click", () => rotatePhysicalPhoto(-90));
$("#rotatePhotoRight").addEventListener("click", () => rotatePhysicalPhoto(90));
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
