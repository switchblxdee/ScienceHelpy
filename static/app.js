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
  if (!response.ok) throw new Error(data.error || tr("Запрос не удался. Попробуй ещё раз."));
  return data;
}

function dateLabel(value) {
  return new Intl.DateTimeFormat(uiLanguage, {day: "numeric", month: "short", timeZone: "UTC"})
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
  return link(tr("Обсудить с ИИ"), "/assistant/" + encodeURIComponent(file), "button outline");
}

function weeklyDiscussion(paper) {
  const form = node("form");
  form.action = "/assistant/from-weekly/" + encodeURIComponent(paper.id);
  form.method = "post";
  form.target = "_blank";
  form.rel = "noopener";
  const submit = node("button", tr("Обсудить с ИИ"), "button outline");
  submit.type = "submit";
  form.append(submit);
  return form;
}

function articleTitle(paper, href) {
  const heading = node("h3");
  heading.append(link(paper.title, href));
  return heading;
}

const votesFormatter = new Intl.PluralRules(uiLanguage);
const votesLabels = {one: "голос", few: "голоса", many: "голосов", other: "голосов"};

function voteLabel(count) {
  const plural = votesFormatter.select(count);
  return uiLanguage === "en" ? (plural === "one" ? "vote" : "votes") : votesLabels[plural];
}

function weeklyCard(paper, entering) {
  const card = node("li", "", entering ? "card enter" : "card");
  const meta = node("div", "", "card-meta");
  meta.append(node("span", "arXiv:" + paper.id, "badge"),
              node("span", `${paper.upvotes} ${voteLabel(paper.upvotes)} · ${paper.comments} ${uiLanguage === "en" ? "comments" : "комм."}`));
  const saved = savedWeekly(paper);
  const save = button(saved ? tr("В библиотеке") : tr("Сохранить"), async () => {
    save.disabled = true;
    save.textContent = tr("Загрузка PDF…");
    try {
      await importWeekly(paper);
      status("weekly-status", "");
    } catch (error) {
      status("weekly-status", error.message);
      save.textContent = tr("Повторить");
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
  description.append(node("p", `${paper.arxiv_id ? "arXiv:" + paper.arxiv_id : tr("Локальный PDF")} · ${tr("{count} стр.", {count: paper.page_count})}`, "card-meta"),
                     articleTitle(paper, pdfUrl(paper.file)));
  if (paper.authors) description.append(node("p", paper.authors, "authors"));
  const actions = node("div", "", "card-actions");
  actions.append(link(tr("Открыть PDF"), pdfUrl(paper.file), "button outline"),
                 discussionLink(paper.file));
  row.append(description, actions);
  return row;
}

function renderCatalog(animateCards = false) {
  const text = filter.value.trim().toLocaleLowerCase();
  const selected = papers.filter((paper) => matches(paper, text));
  const trending = weekly.papers.filter((paper) => matches(paper, text));
  byId("library-status").textContent = libraryError || (!catalogReady
    ? tr("Загружаем статьи…") : text ? tr("Найдено: {count} из {total}", {count: selected.length, total: papers.length})
      : papers.length ? tr("Сохранено статей: {count}", {count: papers.length}) : tr("Здесь будут твои сохранённые статьи."));
  status("weekly-status", weeklyError || (!feedReady ? tr("Загружаем подборку…")
    : !weekly.papers.length ? tr("Подборка ещё обновляется. Загляни чуть позже.")
      : !trending.length ? tr("Совпадений в подборке нет.") : ""));
  if (weekly.start && weekly.end) {
    byId("weekly-period").textContent = `${dateLabel(weekly.start)} — ${dateLabel(weekly.end)} · ${tr("Голоса и обсуждения")}`;
  }
  byId("weekly-list").replaceChildren(...(expanded ? trending : trending.slice(0, 6)).map((paper) => weeklyCard(paper, animateCards)));
  byId("library-list").replaceChildren(...selected.map(libraryRow));
  byId("show-all").hidden = trending.length <= 6;
  byId("show-all").textContent = expanded ? tr("Свернуть подборку") : tr("Вся подборка ({count})", {count: trending.length});
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
  status("upload-status", tr("Читаем PDF…"));
  try {
    const {paper, created} = await api("/api/papers", {method: "POST", body});
    filter.value = "";
    rememberPaper(paper);
    status("upload-status", created ? tr("Сохранено: {title}", {title: paper.title}) : tr("Этот PDF уже в библиотеке."));
  } catch (error) {
    status("upload-status", error.message);
  }
});
const arxivForm = byId("arxiv-import");
arxivForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const reference = byId("arxiv-reference").value.trim();
  if (!reference) return;
  const submit = arxivForm.querySelector("button");
  const result = byId("arxiv-result");
  submit.disabled = true;
  result.hidden = true;
  status("arxiv-status", tr("Скачиваем PDF с arXiv…"));
  try {
    const {paper, created} = await api("/api/arxiv/import", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({reference}),
    });
    filter.value = "";
    rememberPaper(paper);
    status("arxiv-status", created ? tr("Сохранено: {title}", {title: paper.title})
      : tr("Этот PDF уже в библиотеке."));
    const download = link(tr("Скачать PDF"), pdfUrl(paper.file), "button outline");
    download.removeAttribute("target");
    download.download = paper.file;
    result.replaceChildren(link(tr("Открыть PDF"), pdfUrl(paper.file), "button outline"),
                           download, discussionLink(paper.file));
    result.hidden = false;
  } catch (error) {
    status("arxiv-status", error.message);
  } finally {
    submit.disabled = false;
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
