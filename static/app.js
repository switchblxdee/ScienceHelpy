"use strict";

const byId = (id) => document.getElementById(id);
const filter = byId("catalog-filter");
const dialog = byId("chat");
const picker = byId("chat-paper");
const question = byId("question");
const pdfPicker = byId("pdf");
const stream = byId("chat-stream");
const conversations = new Map();
const pendingImports = new Map();
let papers = [];
let weekly = {papers: []};
let libraryError = "";
let weeklyError = "";
let lastFeedCheck = 0;
let catalogReady = false;
let feedReady = false;
let expanded = false;
let activeFile = "";
let selection = 0;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function button(text, callback, className = "button outline") {
  const element = node("button", text, className);
  element.type = "button";
  element.addEventListener("click", callback);
  return element;
}

function link(text, href, className) {
  const element = node("a", text, className);
  element.href = href;
  element.target = "_blank";
  element.rel = "noopener noreferrer";
  return element;
}

function pdfUrl(file, page) {
  return "/papers/" + encodeURIComponent(file) + (page ? "#page=" + Number(page) : "");
}

function arxivUrl(id) {
  return "https://arxiv.org/abs/" + encodeURIComponent(id);
}

function status(id, message) {
  const element = byId(id);
  element.textContent = message;
  element.hidden = !message;
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "Запрос не удался. Попробуй ещё раз.");
  return data;
}

function dateLabel(value) {
  return new Intl.DateTimeFormat("ru", {day: "numeric", month: "short", timeZone: "UTC"})
    .format(new Date(value + "T00:00:00Z"));
}

function matches(paper, text) {
  return [paper.title, paper.authors, paper.arxiv_id, paper.id, paper.file].some(
    (value) => value && value.toLocaleLowerCase().includes(text));
}

function savedWeekly(paper) {
  return papers.find((saved) => saved.arxiv_id === paper.id);
}

function articleTitle(paper, href) {
  const heading = node("h3");
  heading.append(link(paper.title, href));
  return heading;
}

const votesFormatter = new Intl.PluralRules("ru");
const votesLabels = {one: "голос", few: "голоса", many: "голосов", other: "голосов"};

function weeklyCard(paper) {
  const card = node("li", "", "card");
  const meta = node("div", "", "card-meta");
  meta.append(node("span", "arXiv:" + paper.id, "badge"),
              node("span", `${paper.upvotes} ${votesLabels[votesFormatter.select(paper.upvotes)]} · ${paper.comments} комм.`));
  const saved = savedWeekly(paper);
  const save = button(saved ? "В библиотеке" : "Сохранить", async () => {
    save.disabled = true;
    save.textContent = "Загрузка PDF…";
    try {
      await importWeekly(paper);
      status("weekly-status", "");
    } catch (error) {
      status("weekly-status", error.message);
      save.textContent = "Повторить";
      save.disabled = false;
    }
  }, saved ? "button outline saved" : "button outline");
  save.disabled = Boolean(saved);
  const actions = node("div", "", "card-actions");
  actions.append(save, button("Обсудить", () => openChat(saved ? "local:" + saved.file : "weekly:" + paper.id)));
  card.append(meta, articleTitle(paper, arxivUrl(paper.id)));
  if (paper.authors) card.append(node("p", paper.authors, "authors"));
  if (paper.summary) card.append(node("p", paper.summary, "abstract"));
  card.append(actions);
  return card;
}

function libraryRow(paper) {
  const row = node("li", "", "library-row");
  const description = node("div");
  description.append(node("p", `${paper.arxiv_id ? "arXiv:" + paper.arxiv_id : "Локальный PDF"} · ${paper.page_count} стр.`, "card-meta"),
                     articleTitle(paper, pdfUrl(paper.file)));
  if (paper.authors) description.append(node("p", paper.authors, "authors"));
  const actions = node("div", "", "card-actions");
  actions.append(link("Открыть PDF", pdfUrl(paper.file), "button outline"),
                 button("Обсудить", () => openChat("local:" + paper.file)));
  row.append(description, actions);
  return row;
}

function renderCatalog() {
  const text = filter.value.trim().toLocaleLowerCase();
  const selected = papers.filter((paper) => matches(paper, text));
  const trending = weekly.papers.filter((paper) => matches(paper, text));
  byId("library-status").textContent = libraryError || (!catalogReady
    ? "Загружаем статьи…" : text ? `Найдено: ${selected.length} из ${papers.length}`
      : papers.length ? `Сохранено статей: ${papers.length}` : "Здесь будут твои сохранённые статьи.");
  status("weekly-status", weeklyError || (!feedReady ? "Загружаем подборку…"
    : !weekly.papers.length ? "Подборка ещё обновляется. Загляни чуть позже."
      : !trending.length ? "Совпадений в подборке нет." : ""));
  if (weekly.start && weekly.end) {
    byId("weekly-period").textContent = `${dateLabel(weekly.start)} — ${dateLabel(weekly.end)} · Голоса и обсуждения`;
  }
  byId("weekly-list").replaceChildren(...(expanded ? trending : trending.slice(0, 6)).map(weeklyCard));
  byId("library-list").replaceChildren(...selected.map(libraryRow));
  byId("show-all").hidden = trending.length <= 6;
  byId("show-all").textContent = expanded ? "Свернуть подборку" : `Вся подборка (${trending.length})`;
}

function refreshPicker(value = picker.value) {
  const local = node("optgroup");
  local.label = "В библиотеке";
  local.append(...papers.map((paper) => new Option(paper.title, "local:" + paper.file)));
  const trending = node("optgroup");
  trending.label = "Подборка недели";
  trending.append(...weekly.papers.filter((paper) => !savedWeekly(paper))
    .map((paper) => new Option(paper.title, "weekly:" + paper.id)));
  picker.replaceChildren(new Option("Выбери статью", ""));
  if (local.children.length) picker.append(local);
  if (trending.children.length) picker.append(trending);
  if ([...picker.options].some((option) => option.value === value)) picker.value = value;
}

async function loadLibrary() {
  try {
    papers = await api("/api/papers");
    libraryError = "";
  } catch (error) {
    libraryError = error.message;
  } finally {
    catalogReady = true;
    renderCatalog();
    refreshPicker();
  }
}

async function loadWeekly() {
  lastFeedCheck = Date.now();
  try {
    weekly = await api("/api/weekly");
    weeklyError = "";
  } catch (error) {
    weeklyError = error.message;
  } finally {
    feedReady = true;
    renderCatalog();
    refreshPicker();
  }
}

function rememberPaper(paper) {
  papers = [...papers.filter((item) => item.file !== paper.file), paper];
  renderCatalog();
  refreshPicker();
}

async function importWeekly(paper) {
  const saved = savedWeekly(paper);
  if (saved) return saved;
  if (!pendingImports.has(paper.id)) {
    const promise = api(`/api/weekly/${encodeURIComponent(paper.id)}/save`, {method: "POST"})
      .then(({paper: stored}) => {
        rememberPaper(stored);
        return stored;
      })
      .finally(() => pendingImports.delete(paper.id));
    pendingImports.set(paper.id, promise);
  }
  return pendingImports.get(paper.id);
}

function conversation(file) {
  if (!conversations.has(file)) conversations.set(file, []);
  return conversations.get(file);
}

function bubble(role, text, data) {
  const item = node("div", "", "bubble " + role);
  item.append(node("p", text));
  if (data?.warning) item.append(node("p", data.warning, "notice"));
  if (data?.sources.length) {
    const sources = node("ol", "", "source-list");
    for (const source of data.sources) {
      const row = node("li");
      const details = node("details");
      const label = source.page ? `Страница ${source.page} · ${source.title}` : source.title;
      details.append(node("summary", label), node("p", source.text));
      if (source.file) details.append(link("Открыть страницу PDF", pdfUrl(source.file, source.page)));
      row.append(details);
      sources.append(row);
    }
    item.append(sources);
  }
  return item;
}

function renderConversation() {
  const turns = activeFile ? conversation(activeFile) : [];
  byId("chat-empty").hidden = turns.length > 0;
  byId("chat-intro").textContent = activeFile
    ? "Спроси о методе, результатах или о том, где авторы обсуждают нужную тему."
    : "Выбери статью, чтобы начать диалог.";
  byId("chat-suggestions").hidden = !activeFile;
  const messages = byId("chat-messages");
  messages.replaceChildren(...turns.flatMap((turn) => [
    bubble("user", turn.question),
    bubble("assistant", turn.data?.answer || turn.error || "Ищу ответ в статье…", turn.data),
  ]));
  question.disabled = !activeFile || Boolean(turns.at(-1)?.pending);
  byId("send").disabled = question.disabled;
  stream.scrollTop = stream.scrollHeight;
}

async function selectPaper(value) {
  const request = ++selection;
  activeFile = "";
  status("chat-status", "");
  renderConversation();
  question.value = "";
  if (value.startsWith("local:")) {
    activeFile = value.slice(6);
  } else if (value.startsWith("weekly:")) {
    const paper = weekly.papers.find((item) => item.id === value.slice(7));
    if (!paper) return;
    status("chat-status", "Загружаю PDF с arXiv…");
    try {
      const saved = await importWeekly(paper);
      if (request !== selection) return;
      refreshPicker("local:" + saved.file);
      activeFile = saved.file;
      status("chat-status", "");
    } catch (error) {
      if (request !== selection) return;
      picker.value = "";
      status("chat-status", error.message);
    }
  }
  renderConversation();
  if (activeFile && dialog.open) question.focus();
}

function openChat(value) {
  if (!dialog.open) dialog.showModal();
  if (value) {
    refreshPicker(value);
    selectPaper(value);
  } else {
    renderConversation();
    picker.focus();
  }
}

async function ask(text) {
  if (!activeFile || text.length < 3 || question.disabled) return;
  const file = activeFile;
  const turns = conversation(file);
  const history = turns.filter((turn) => turn.data).slice(-3)
    .map((turn) => ({question: turn.question, answer: turn.data.answer.slice(0, 2500)}));
  const turn = {question: text, pending: true};
  if (turns.length >= 40) turns.shift();
  turns.push(turn);
  question.value = "";
  renderConversation();
  try {
    turn.data = await api("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: text, file, history}),
    });
  } catch (error) {
    turn.error = error.message;
  } finally {
    turn.pending = false;
    if (file === activeFile) {
      renderConversation();
      if (dialog.open) question.focus();
    }
  }
}

filter.addEventListener("input", renderCatalog);
byId("show-all").addEventListener("click", () => {
  expanded = !expanded;
  renderCatalog();
});
picker.addEventListener("change", () => selectPaper(picker.value));
for (const item of document.querySelectorAll("[data-open-chat]")) {
  item.addEventListener("click", () => openChat());
}
for (const item of document.querySelectorAll("[data-question]")) {
  item.addEventListener("click", () => ask(item.dataset.question));
}
for (const item of document.querySelectorAll("[data-upload]")) {
  item.addEventListener("click", () => pdfPicker.click());
}
pdfPicker.addEventListener("change", async () => {
  if (!pdfPicker.files.length) return;
  const body = new FormData();
  body.append("pdf", pdfPicker.files[0]);
  pdfPicker.value = "";
  status("upload-status", "Читаем PDF…");
  try {
    const {paper, created} = await api("/api/papers", {method: "POST", body});
    filter.value = "";
    rememberPaper(paper);
    status("upload-status", created ? `Сохранено: ${paper.title}` : "Этот PDF уже в библиотеке.");
  } catch (error) {
    status("upload-status", error.message);
  }
});
byId("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  ask(question.value.trim());
});
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    byId("chat-form").requestSubmit();
  }
});

loadLibrary();
loadWeekly();

const feedCheckInterval = 60 * 60 * 1000;
function refreshVisibleFeed() {
  if (!document.hidden && Date.now() - lastFeedCheck >= feedCheckInterval) loadWeekly();
}
document.addEventListener("visibilitychange", refreshVisibleFeed);
setInterval(refreshVisibleFeed, feedCheckInterval);
