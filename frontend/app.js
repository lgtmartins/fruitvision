/*
 * FruitVision - frontend
 * Fala com o backend Flask (backend/app.py) via fetch(). Não há build step:
 * basta servir esta pasta estaticamente.
 */

const API_BASE = "http://localhost:5000";

const FRUITS = {
  apple:     { pt: "Maçã",    emoji: "🍎", color: "#E5484D" },
  avocado:   { pt: "Abacate", emoji: "🥑", color: "#65A30D" },
  banana:    { pt: "Banana",  emoji: "🍌", color: "#EAB308" },
  guava:     { pt: "Goiaba",  emoji: "🍈", color: "#10B981" },
  kiwi:      { pt: "Kiwi",    emoji: "🥝", color: "#84CC16" },
  mango:     { pt: "Manga",   emoji: "🥭", color: "#F59E0B" },
  orange:    { pt: "Laranja", emoji: "🍊", color: "#F97316" },
  peach:     { pt: "Pêssego", emoji: "🍑", color: "#F472B6" },
  pineapple: { pt: "Abacaxi", emoji: "🍍", color: "#CA8A04" },
};

function fruit(className) {
  return FRUITS[className] || { pt: className, emoji: "🍽️", color: "#64748B" };
}

const $ = (id) => document.getElementById(id);

/** Cria um elemento com classe, texto e estilos opcionais (evita innerHTML com dados do usuário). */
function el(tag, { className, text, style } = {}, children = []) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (style) Object.entries(style).forEach(([k, v]) => node.style.setProperty(k, v));
  children.forEach((child) => node.append(child));
  return node;
}

const pct = (value) => `${Math.round(value * 100)}%`;

function pluralFruits(n) {
  if (n === 0) return "Nenhuma fruta detectada";
  return n === 1 ? "1 fruta detectada" : `${n} frutas detectadas`;
}

/** Conta quantas detecções há de cada classe, da mais frequente para a menos. */
function countByClass(items) {
  const counts = {};
  items.forEach((i) => { counts[i.class_name] = (counts[i.class_name] || 0) + 1; });
  return Object.entries(counts).sort((a, b) => b[1] - a[1]);
}

function fruitChip(className, count) {
  const f = fruit(className);
  const chip = el("span", { className: "fruit-chip", style: { "--c": f.color } }, [
    el("span", { className: "emoji", text: f.emoji }),
    f.pt,
  ]);
  if (count !== undefined) chip.append(el("b", { text: `×${count}` }));
  return chip;
}

function detectionList(listEl, items) {
  listEl.replaceChildren();
  [...items]
    .sort((a, b) => b.confidence - a.confidence)
    .forEach((det) => {
      const f = fruit(det.class_name);
      listEl.append(
        el("li", { className: "detection", style: { "--c": f.color } }, [
          el("span", { className: "emoji", text: f.emoji }),
          el("span", { className: "name", text: f.pt }),
          el("span", { className: "pct", text: pct(det.confidence) }),
          el("span", { className: "bar" }, [el("span", { style: { width: pct(det.confidence) } })]),
        ])
      );
    });
  if (!items.length) {
    listEl.append(el("li", {
      className: "no-detections",
      text: "Nenhuma fruta encontrada. Tente diminuir a confiança mínima.",
    }));
  }
}

function formatDate(iso) {
  const date = new Date(iso);
  const diffSec = (date - new Date()) / 1000;
  const rtf = new Intl.RelativeTimeFormat("pt-BR", { numeric: "auto" });
  const steps = [[60, "second"], [3600, "minute"], [86400, "hour"], [604800, "day"]];
  for (let i = 0; i < steps.length; i++) {
    const [limit] = steps[i];
    if (Math.abs(diffSec) < limit) {
      const divisor = i === 0 ? 1 : steps[i - 1][0];
      return rtf.format(Math.round(diffSec / divisor), steps[i][1]);
    }
  }
  return date.toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" });
}

/* ---------------- tema claro/escuro ---------------- */

/*
 * O tema inicial é aplicado por um script no <head> do index.html, para a
 * página não piscar branco ao abrir no modo escuro. Aqui só tratamos o botão.
 * Sem escolha salva, o site acompanha a configuração do sistema.
 */

const TEMA_CHAVE = "fruitvision-tema";
const themeBtn = $("theme-btn");

function temaSalvo() {
  try {
    return localStorage.getItem(TEMA_CHAVE);
  } catch (e) {
    return null; // navegador com armazenamento bloqueado
  }
}

function temaDoSistema() {
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function aplicarTema(tema) {
  document.documentElement.dataset.theme = tema;
  const proximo = tema === "dark" ? "claro" : "escuro";
  themeBtn.title = `Mudar para o tema ${proximo}`;
  themeBtn.setAttribute("aria-label", `Mudar para o tema ${proximo}`);
}

themeBtn.addEventListener("click", () => {
  const novo = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  aplicarTema(novo);
  try {
    localStorage.setItem(TEMA_CHAVE, novo);
  } catch (e) {
    /* sem armazenamento: o tema vale só até recarregar a página */
  }
});

// enquanto o usuário não escolher, segue o sistema mesmo se ele mudar com a página aberta
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (!temaSalvo()) aplicarTema(temaDoSistema());
});

aplicarTema(temaSalvo() || temaDoSistema());

/* ---------------- abas ---------------- */

const TABS = {
  detect:  { panel: $("panel-detect"),  hash: "analisar" },
  history: { panel: $("panel-history"), hash: "historico", onOpen: () => loadHistory() },
  stats:   { panel: $("panel-stats"),   hash: "estatisticas", onOpen: () => loadStats() },
};
const tabButtons = document.querySelectorAll(".tab-btn");

function openTab(name) {
  const tab = TABS[name] || TABS.detect;
  tabButtons.forEach((b) => {
    const active = b.dataset.tab === name;
    b.classList.toggle("is-active", active);
    b.setAttribute("aria-selected", active);
  });
  Object.values(TABS).forEach((t) => t.panel.classList.toggle("is-active", t === tab));
  history.replaceState(null, "", `#${tab.hash}`);
  tab.onOpen?.();
}

tabButtons.forEach((btn) => btn.addEventListener("click", () => openTab(btn.dataset.tab)));

/* ---------------- upload / preview ---------------- */

const dropzone = $("dropzone");
const fileInput = $("file-input");
const preview = $("preview");
const previewImg = $("preview-img");
const analyzeBtn = $("analyze-btn");
const confSlider = $("conf-slider");

let selectedFile = null;
let previewUrl = null;

$("species-list").append(
  ...Object.keys(FRUITS).map((key) => el("li", {}, [fruitChip(key)]))
);

function updateSlider() {
  const { min, max, value } = confSlider;
  confSlider.style.setProperty("--fill", `${((value - min) / (max - min)) * 100}%`);
  $("conf-value").textContent = pct(Number(value));
}
confSlider.addEventListener("input", updateSlider);
updateSlider();

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

// Ctrl+V com uma imagem copiada
document.addEventListener("paste", (e) => {
  const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (file) {
    openTab("detect");
    handleFile(file);
  }
});

function handleFile(file) {
  const validTypes = ["image/jpeg", "image/jpg", "image/png"];
  if (!validTypes.includes(file.type)) {
    showError("Formato não suportado. Envie um arquivo .jpg, .jpeg ou .png.");
    return;
  }
  // imagens coladas chegam como "image.png"; o backend precisa de um nome com extensão
  selectedFile = file.name ? file : new File([file], "imagem-colada.png", { type: file.type });
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImg.src = previewUrl;
  preview.hidden = false;
  dropzone.hidden = true;
  analyzeBtn.disabled = false;
  showState("empty");
}

$("clear-btn").addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  preview.hidden = true;
  dropzone.hidden = false;
  analyzeBtn.disabled = true;
  showState("empty");
});

/* ---------------- análise ---------------- */

const STATES = {
  empty: $("result-empty"),
  loading: $("result-loading"),
  error: $("result-error"),
  content: $("result-content"),
};

function showState(name) {
  Object.entries(STATES).forEach(([key, node]) => { node.hidden = key !== name; });
}

function showError(message) {
  $("result-error-msg").textContent = message;
  showState("error");
}

function connectionError(err) {
  return err instanceof TypeError
    ? `Não foi possível conectar ao backend. Verifique se ele está rodando em ${API_BASE}.`
    : err.message;
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  $("loading-img").src = previewUrl;
  showState("loading");
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("conf", confSlider.value);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Falha ao analisar a imagem.");
    await renderResult(data);
  } catch (err) {
    showError(connectionError(err));
  } finally {
    analyzeBtn.disabled = false;
  }
});

async function renderResult(data) {
  const img = $("result-img");
  img.src = `${API_BASE}${data.annotated_image_url}`;
  await img.decode().catch(() => {});

  $("result-count").textContent = pluralFruits(data.total_objects);
  $("result-tags").replaceChildren(
    ...countByClass(data.detections).map(([cls, n]) => fruitChip(cls, n))
  );
  detectionList($("result-list"), data.detections);
  showState("content");

  // no celular o resultado fica abaixo do formulário
  if (matchMedia("(max-width: 860px)").matches) {
    document.querySelector(".result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/* ---------------- histórico ---------------- */

const historyGrid = $("history-grid");
const historyEmpty = $("history-empty");

$("refresh-history-btn").addEventListener("click", loadHistory);

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/history?limit=60`);
    const items = await res.json();

    historyGrid.replaceChildren();
    historyEmpty.hidden = items.length > 0;
    $("history-sub").textContent = items.length
      ? `${items.length} ${items.length === 1 ? "análise" : "análises"} · clique em uma para ver os detalhes.`
      : "Suas análises mais recentes.";
    resetEmptyText(historyEmpty, "Nenhuma análise ainda", "As imagens que você analisar vão aparecer aqui.");

    items.forEach((item) => {
      const emojis = countByClass(item.items).map(([cls]) => fruit(cls).emoji).join("");
      const img = el("img", {});
      img.src = `${API_BASE}/api/images/${item.annotated_image_path}`;
      img.alt = `Resultado de ${item.original_filename}`;
      img.loading = "lazy";

      const card = el("button", { className: "card history-card" }, [
        img,
        el("div", { className: "history-body" }, [
          el("div", { className: "history-emojis" }, [
            emojis ? emojis : el("span", { className: "none", text: "Nenhuma fruta" }),
          ]),
          el("p", { className: "history-title", text: item.original_filename }),
          el("p", {
            className: "history-meta",
            text: `${pluralFruits(item.items.length)} · ${formatDate(item.created_at)}`,
          }),
        ]),
      ]);
      card.type = "button";
      card.addEventListener("click", () => openDetail(item));
      historyGrid.append(card);
    });
  } catch (err) {
    historyGrid.replaceChildren();
    historyEmpty.hidden = false;
    resetEmptyText(historyEmpty, "Não foi possível carregar o histórico", "O backend está rodando?");
  }
}

function resetEmptyText(container, title, sub) {
  container.querySelector(".state-title").textContent = title;
  container.querySelector(".state-sub").textContent = sub;
}

/* ---------------- modal de detalhe ---------------- */

const modal = $("detail-modal");
let currentDetail = null;

function openDetail(item) {
  currentDetail = item;
  $("detail-title").textContent = item.original_filename;
  $("detail-date").textContent = new Date(item.created_at).toLocaleString("pt-BR", {
    dateStyle: "long",
    timeStyle: "short",
  });
  $("detail-img").src = `${API_BASE}/api/images/${item.annotated_image_path}`;
  $("detail-img").alt = `Detecções em ${item.original_filename}`;
  detectionList($("detail-list"), item.items);
  modal.showModal();
}

$("detail-close").addEventListener("click", () => modal.close());
modal.addEventListener("click", (e) => { if (e.target === modal) modal.close(); });

$("detail-delete").addEventListener("click", async () => {
  if (!currentDetail || !confirm("Excluir esta análise do histórico?")) return;
  try {
    const res = await fetch(`${API_BASE}/api/history/${currentDetail.id}`, { method: "DELETE" });
    if (!res.ok) throw new Error((await res.json()).error || "Falha ao excluir.");
    modal.close();
    loadHistory();
  } catch (err) {
    alert(connectionError(err));
  }
});

/* ---------------- estatísticas ---------------- */

async function loadStats() {
  const empty = $("stats-empty");
  const content = $("stats-content");
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    const stats = await res.json();

    resetEmptyText(empty, "Ainda não há dados", "Analise algumas imagens para ver as estatísticas.");
    empty.hidden = stats.length > 0;
    content.hidden = stats.length === 0;
    if (!stats.length) return;

    const total = stats.reduce((sum, s) => sum + s.total, 0);
    const weightedConf = stats.reduce((sum, s) => sum + s.avg_confidence * s.total, 0) / total;
    $("kpi-total").textContent = total.toLocaleString("pt-BR");
    $("kpi-species").textContent = `${stats.length} de ${Object.keys(FRUITS).length}`;
    $("kpi-conf").textContent = pct(weightedConf);

    const max = Math.max(...stats.map((s) => s.total));
    $("stats-list").replaceChildren(
      ...stats.map((s) => {
        const f = fruit(s.class_name);
        return el("div", { className: "stats-row" }, [
          el("span", { className: "stats-fruit" }, [el("span", { className: "emoji", text: f.emoji }), f.pt]),
          el("span", { className: "stats-bar", style: { "--c": f.color } }, [
            el("span", { className: "bar" }, [el("span", { style: { width: `${(s.total / max) * 100}%` } })]),
            el("span", { className: "count", text: s.total }),
          ]),
          el("span", { className: "num", text: pct(s.avg_confidence) }),
        ]);
      })
    );
  } catch (err) {
    content.hidden = true;
    empty.hidden = false;
    resetEmptyText(empty, "Não foi possível carregar as estatísticas", "O backend está rodando?");
  }
}

/* ---------------- início ---------------- */

const tabFromHash = () => Object.keys(TABS).find((k) => `#${TABS[k].hash}` === location.hash) || "detect";
window.addEventListener("hashchange", () => openTab(tabFromHash()));
openTab(tabFromHash());
