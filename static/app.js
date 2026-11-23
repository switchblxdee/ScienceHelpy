"use strict";

const byId = (id) => document.getElementById(id);
const filter = byId("catalog-filter");
const pdfPicker = byId("pdf");
const pendingImports = new Map();
let papers = [];
let weekly = {papers: []};
let libraryError = "";
let weeklyError = "";
let lastFeedCheck = 0;
let catalogReady = false;
let feedReady = false;
let expanded = false;

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

function discussionLink(file) {
  return link("Обсудить с ИИ", "/assistant/" + encodeURIComponent(file), "button outline");
}

function weeklyDiscussion(paper) {
  const form = node("form");
  form.action = "/assistant/from-weekly/" + encodeURIComponent(paper.id);
  form.method = "post";
  form.target = "_blank";
  form.rel = "noopener";
  const submit = node("button", "Обсудить с ИИ", "button outline");
  submit.type = "submit";
  form.append(submit);
  return form;
}

function articleTitle(paper, href) {
  const heading = node("h3");
  heading.append(link(paper.title, href));
  return heading;
}

const votesFormatter = new Intl.PluralRules("ru");
const votesLabels = {one: "голос", few: "голоса", many: "голосов", other: "голосов"};

function weeklyCard(paper, entering) {
  const card = node("li", "", entering ? "card enter" : "card");
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
  actions.append(save, saved ? discussionLink(saved.file) : weeklyDiscussion(paper));
  const record = node("div", "", "record");
  record.append(articleTitle(paper, arxivUrl(paper.id)), meta);
  if (paper.authors) record.append(node("p", paper.authors, "authors"));
  if (paper.summary) record.append(node("p", paper.summary, "abstract"));
  card.append(record, actions);
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
                 discussionLink(paper.file));
  row.append(description, actions);
  return row;
}

function renderCatalog(animateCards = false) {
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
  byId("weekly-list").replaceChildren(...(expanded ? trending : trending.slice(0, 6)).map((paper) => weeklyCard(paper, animateCards)));
  byId("library-list").replaceChildren(...selected.map(libraryRow));
  byId("show-all").hidden = trending.length <= 6;
  byId("show-all").textContent = expanded ? "Свернуть подборку" : `Вся подборка (${trending.length})`;
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
    renderCatalog(true);
  }
}

function rememberPaper(paper) {
  papers = [...papers.filter((item) => item.file !== paper.file), paper];
  renderCatalog();
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

filter.addEventListener("input", () => renderCatalog());
byId("show-all").addEventListener("click", () => {
  expanded = !expanded;
  renderCatalog(expanded);
});
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
loadLibrary();
loadWeekly();

const feedCheckInterval = 60 * 60 * 1000;
function refreshVisibleFeed() {
  if (!document.hidden && Date.now() - lastFeedCheck >= feedCheckInterval) loadWeekly();
}
document.addEventListener("visibilitychange", refreshVisibleFeed);
setInterval(refreshVisibleFeed, feedCheckInterval);
