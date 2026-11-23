"use strict";

const byId = (id) => document.getElementById(id);
const filter = byId("filter");
const list = byId("papers");
const scope = byId("scope");
const question = byId("question");
const result = byId("result");
let papers = [];

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "Запрос не удался. Попробуйте ещё раз.");
  return data;
}

function pdfUrl(file, page) {
  return "/papers/" + encodeURIComponent(file) + (page ? "#page=" + Number(page) : "");
}

function link(text, href) {
  const node = element("a", text);
  node.href = href;
  node.target = "_blank";
  node.rel = "noopener noreferrer";
  return node;
}

function renderPapers() {
  const query = filter.value.trim().toLocaleLowerCase();
  const matches = papers.filter((paper) =>
    [paper.title, paper.authors, paper.arxiv_id, paper.file].some((field) =>
      field.toLocaleLowerCase().includes(query)));
  byId("catalog-status").textContent = query
    ? `Найдено: ${matches.length} из ${papers.length}`
    : papers.length ? `В библиотеке: ${papers.length}` : "Библиотека пуста. Добавьте PDF.";
  list.replaceChildren(...matches.map((paper) => {
    const row = element("li", "", "paper");
    const meta = element("div", "", "paper-meta");
    meta.append(element("span", paper.arxiv_id ? `arXiv:${paper.arxiv_id}` : "ЛОКАЛЬНЫЙ PDF", "tag"),
      element("span", `· ${paper.page_count} стр.`));
    const title = element("h3");
    title.append(link(paper.title, pdfUrl(paper.file)));
    const actions = element("div", "", "paper-actions");
    actions.append(link("Открыть PDF", pdfUrl(paper.file)));
    const ask = element("button", "Спросить по статье", "text-button");
    ask.type = "button";
    ask.addEventListener("click", () => {
      scope.value = paper.file;
      question.focus();
    });
    actions.append(ask);
    row.append(meta, title);
    if (paper.authors) row.append(element("p", paper.authors, "author"));
    row.append(actions);
    return row;
  }));
}

async function refreshPapers() {
  papers = await api("/api/papers");
  const selected = scope.value;
  scope.replaceChildren(new Option("Во всей библиотеке", ""),
    ...papers.map((paper) => new Option(paper.title, paper.file)));
  if (papers.some((paper) => paper.file === selected)) scope.value = selected;
  renderPapers();
}

function renderAnswer(data) {
  result.replaceChildren(element("p", data.answer));
  if (data.warning) result.append(element("p", data.warning, "status"));
  if (!data.sources.length) return;
  const sources = element("ol", "", "source-list");
  for (const source of data.sources) {
    const item = element("li");
    const detail = element("details");
    const label = source.page ? `${source.title} · стр. ${source.page}` : source.title;
    detail.append(element("summary", label), element("p", source.text));
    if (source.file) detail.append(link(`Страница ${source.page} в PDF`, pdfUrl(source.file, source.page)));
    else if (/^https?:\/\//i.test(source.url)) detail.append(link("Открыть источник", source.url));
    item.append(detail);
    sources.append(item);
  }
  result.append(sources);
}

filter.addEventListener("input", renderPapers);
for (const chip of document.querySelectorAll("[data-example]")) {
  chip.addEventListener("click", () => {
    scope.value = "";
    question.value = chip.dataset.example;
    question.focus();
  });
}

const picker = byId("pdf");
const uploadButton = byId("upload-button");
uploadButton.addEventListener("click", () => picker.click());
picker.addEventListener("change", async () => {
  if (!picker.files.length) return;
  const status = byId("upload-status");
  const body = new FormData();
  body.append("pdf", picker.files[0]);
  uploadButton.disabled = true;
  status.hidden = false;
  status.textContent = "Читаем PDF…";
  try {
    const data = await api("/api/papers", {method: "POST", body});
    filter.value = "";
    await refreshPapers();
    status.textContent = data.created ? `Добавлено: ${data.paper.title}` : "Этот PDF уже в библиотеке.";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
    picker.value = "";
  }
});

byId("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("ask-button");
  if (button.disabled) return;
  const prompt = question.value.trim();
  if (prompt.length < 3) {
    result.hidden = false;
    result.textContent = "Введите вопрос длиной от трёх символов.";
    return;
  }
  button.disabled = true;
  button.textContent = "Ищем источники…";
  result.hidden = false;
  result.textContent = "Ищем подходящие фрагменты…";
  try {
    const data = await api("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: prompt, file: scope.value || null}),
    });
    renderAnswer(data);
  } catch (error) {
    result.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Найти по статьям";
  }
});

refreshPapers().catch((error) => { byId("catalog-status").textContent = error.message; });
