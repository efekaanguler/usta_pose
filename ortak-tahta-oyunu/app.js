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
const BOARD_TYPES = [
  ["A", "B", "A", "N", "B", "N"],
  ["B", "N", "B", "A", "N", "A"],
  ["N", "A", "N", "B", "A", "B"]
];
const TARGETS = {
  circles: { cell: "A", primary: "Daire", primaryPoints: 4, secondary: "Kare", secondaryPoints: 3 },
  polygons: { cell: "B", primary: "Altıgen", primaryPoints: 4, secondary: "Yıldız", secondaryPoints: 3 }
};
const STORAGE_KEY = "usta-ortak-tahta-v1";

let state = null;
let setupScenario = 1;
let tradeSelection = { P1: null, P2: null };
let toastTimer = null;
let audioContext = null;

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

function targetHtml(targetKey) {
  const target = TARGETS[targetKey];
  return `<small>GİZLİ HEDEF</small><strong><b>${target.cell}</b> hücresindeki her ${target.primary}: +${target.primaryPoints}</strong><strong><b>${target.cell}</b> hücresindeki her ${target.secondary}: +${target.secondaryPoints}</strong>`;
}

function diskHtml(disk, extraClass = "") {
  const light = LIGHT_COLORS.has(disk.color) ? " light" : "";
  return `<button class="disk-button${light} ${extraClass}" data-disk="${disk.id}" type="button" style="background:${COLOR_VALUES[disk.color]}" aria-label="${disk.color}, ${disk.number}, ${disk.shape}"><span class="disk-number">${disk.number}</span><span class="disk-shape" aria-hidden="true">${SHAPE_ICONS[disk.shape]}</span></button>`;
}

function renderBoard() {
  const board = $("#board");
  board.innerHTML = state.board.map((placement, index) => {
    const row = Math.floor(index / 6);
    const number = index % 6 + 1;
    const type = BOARD_TYPES[row][number - 1];
    const selected = state.selectedDisk ? DISK_MAP[state.selectedDisk] : null;
    const eligible = selected && !placement && selected.number === number;
    let content = "";
    if (placement) {
      const disk = DISK_MAP[placement.diskId];
      const light = LIGHT_COLORS.has(disk.color) ? " light" : "";
      content = `<div class="placed-disk${light}" style="background:${COLOR_VALUES[disk.color]}" title="${disk.color} · ${disk.number}/${disk.shape}"><strong>${disk.number}</strong><small>${SHAPE_ICONS[disk.shape]}</small></div><span class="owner-chip ${placement.owner.toLowerCase()}">${placement.owner}</span>`;
    }
    return `<button class="board-cell type-${type}${eligible ? " eligible" : ""}" data-cell="${index}" type="button" aria-label="Satır ${row + 1}, ${number}${type}${placement ? `, ${DISK_MAP[placement.diskId].color} disk, ${placement.owner}` : ", boş"}"><span class="cell-meta"><b>${number}</b><small>${type}</small></span>${content}</button>`;
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
  $("#targetP1").innerHTML = targetHtml(state.targets.P1);
  $("#targetP2").innerHTML = targetHtml(state.targets.P2);
  ["P1", "P2"].forEach(player => {
    const active = state.activePlayer === player;
    $(`#playerPanel${player.slice(1)}`).classList.toggle("active", active);
    $(`#playerPanel${player.slice(1)} .select-player`).textContent = active ? "Aktif" : "Seç";
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
  state.activePlayer = player;
  state.selectedDisk = null;
  $$(".secret-card").forEach(card => card.classList.remove("revealed"));
  sound("tap");
  render();
}

function selectDisk(id) {
  if (!state.hands[state.activePlayer].includes(id)) return;
  state.selectedDisk = state.selectedDisk === id ? null : id;
  sound("tap");
  render();
}

function placeDisk(cellIndex) {
  if (!state.selectedDisk || state.board[cellIndex]) return;
  const disk = DISK_MAP[state.selectedDisk];
  const cellNumber = cellIndex % 6 + 1;
  if (disk.number !== cellNumber) {
    showToast(`Bu hücre ${cellNumber} numaralı bir disk bekliyor.`);
    sound("error");
    return;
  }
  state.history.push({
    type: "place",
    player: state.activePlayer,
    diskId: disk.id,
    cellIndex
  });
  state.board[cellIndex] = { diskId: disk.id, owner: state.activePlayer };
  state.hands[state.activePlayer] = state.hands[state.activePlayer].filter(id => id !== disk.id);
  state.selectedDisk = null;
  sound("place");
  render();
  const filled = state.board.filter(Boolean).length;
  if (filled === 18) {
    showToast("Tahta tamamlandı — iki oyuncuya da +20!");
    setTimeout(() => openScore(true), 550);
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
    if (!placement || placement.owner !== player) return total;
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
  $("#scoreTitle").textContent = complete ? "Tahta tamamlandı!" : "Oyun devam ediyor";
  $("#scoreNameP1").textContent = state.names.P1;
  $("#scoreNameP2").textContent = state.names.P2;
  $("#scoreP1").textContent = p1Target + bonus;
  $("#scoreP2").textContent = p2Target + bonus;
  $("#scoreDetailP1").textContent = `Hedef ${p1Target} ${complete ? "+ ortak 20" : "· bonus kilitli"}`;
  $("#scoreDetailP2").textContent = `Hedef ${p2Target} ${complete ? "+ ortak 20" : "· bonus kilitli"}`;
  $("#completionBonus").classList.toggle("earned", complete);
  $("#completionBonus small").textContent = complete ? "Kazanıldı — iki skora da eklendi." : "18 hücre dolduğunda iki oyuncuya da eklenir.";
  $("#scoreDialog").showModal();
  if (forceReveal) sound("finish");
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
$("#rulesButton").addEventListener("click", () => $("#rulesDialog").showModal());
$("#tradeButton").addEventListener("click", openTrade);
$("#confirmTrade").addEventListener("click", confirmTrade);
$("#undoButton").addEventListener("click", undo);
$("#scoreButton").addEventListener("click", () => openScore(false));
$("#newGameButton").addEventListener("click", newGame);

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
