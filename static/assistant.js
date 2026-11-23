"use strict";

const byId = (id) => document.getElementById(id);
const workspace = byId("workspace");
const question = byId("question");
const chatScroll = byId("chat-scroll");
const paper = encodeURIComponent(workspace.dataset.paper);
const apiBase = "/api/chats/" + paper;
const pdfBase = "/papers/" + paper;
const pageCount = Number(workspace.dataset.pages);
const pending = new Map();
let threads = [];
let active = null;
let selection = 0;
let loading = false;
let creating = false;
let initializing = true;
let page = 1;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function message(id, value) {
  const element = byId(id);
  element.textContent = value;
  element.hidden = !value;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const failure = new Error(body.error || "Запрос не удался. Попробуй ещё раз.");
    failure.status = response.status;
    throw failure;
  }
  return body;
}

function setPanel(panel) {
  workspace.dataset.activePanel = panel;
  for (const control of document.querySelectorAll("[data-panel]")) {
    control.setAttribute("aria-pressed", String(control.dataset.panel === panel));
  }
}

function showPage(number, fromChat = false) {
  if (!Number.isInteger(number) || number < 1 || number > pageCount) {
    byId("page-number").value = page;
    return;
  }
  if (number !== page) {
    page = number;
    const image = byId("pdf-page");
    image.hidden = false;
    byId("page-error").hidden = true;
    image.alt = `Страница ${page} из PDF`;
    image.src = `${pdfBase}/page/${page}.png`;
    byId("document-scroll").scrollTop = 0;
  }
  byId("page-number").value = page;
  byId("previous-page").disabled = page <= 1;
  byId("next-page").disabled = page >= pageCount;
  byId("open-pdf").href = `${pdfBase}#page=${page}`;
  if (fromChat) {
    setPanel("document");
    byId("page-number").focus({preventScroll: true});
  }
}

function renderThreadList() {
  const items = threads.map((thread) => {
    const row = node("li");
    const link = node("a", thread.title, "thread-link");
    link.href = "?chat=" + thread.id;
    link.title = thread.title;
    if (thread.id === active?.id) link.setAttribute("aria-current", "true");
    link.addEventListener("click", (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
      event.preventDefault();
      setPanel("conversation");
      loadThread(thread.id, true);
    });
    row.append(link);
    return row;
  });
  byId("thread-list").replaceChildren(...items);
  message("threads-status", threads.length ? "" : "Здесь появятся чаты по этой статье.");
}

function bubble(role, text) {
  const element = node("div", "", "bubble " + role);
  element.append(node("p", text));
  return element;
}

function replyBubble(turn) {
  const reply = bubble("reply", turn.answer);
  if (turn.warning) reply.append(node("p", turn.warning, "notice"));
  if (!turn.sources.length) return reply;
  const sources = node("ol", "", "source-list");
  for (const source of turn.sources) {
    if (!Number.isInteger(source.page)) continue;
    const item = node("li");
    const detail = node("details");
    const open = node("button", "Показать в PDF", "source-page");
    open.type = "button";
    open.addEventListener("click", () => showPage(source.page, true));
    detail.append(node("summary", `Страница ${source.page}`), node("p", source.text), open);
    item.append(detail);
    sources.append(item);
  }
  if (sources.children.length) reply.append(sources);
  return reply;
}

function renderConversation() {
  const turns = active?.turns ?? [];
  const current = active && pending.get(active.id);
  byId("welcome").hidden = loading || turns.length > 0 || Boolean(current);
  const items = turns.flatMap((turn) => [bubble("user", turn.question), replyBubble(turn)]);
  if (loading) items.push(bubble("reply", "Загружаем историю…"));
  if (current) items.push(bubble("user", current), bubble("reply", "Ищем ответ в статье…"));
  byId("messages").replaceChildren(...items);
  question.disabled = loading || creating || Boolean(current);
  byId("send").disabled = question.disabled;
  byId("new-chat").disabled = creating || initializing;
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function updateUrl(thread, push = false) {
  const url = new URL(window.location.href);
  if (thread) url.searchParams.set("chat", thread.id);
  else url.searchParams.delete("chat");
  if (url.href === window.location.href) return;
  if (push) history.pushState(null, "", url);
  else history.replaceState(null, "", url);
}

async function loadThread(id, push = false) {
  const request = ++selection;
  message("chat-error", "");
  loading = Boolean(id);
  active = null;
  renderConversation();
  try {
    const thread = id ? await api(apiBase + "/" + encodeURIComponent(id)) : null;
    if (request !== selection) return;
    active = thread;
    updateUrl(thread, push);
    question.value = "";
  } catch (error) {
    if (request === selection) message("chat-error", error.message);
  } finally {
    if (request === selection) {
      loading = false;
      renderThreadList();
      renderConversation();
    }
  }
}

function rememberThread(thread) {
  const summary = {id: thread.id, title: thread.title, updated_at: thread.updated_at};
  threads = [summary, ...threads.filter((item) => item.id !== thread.id)];
}

async function createThread() {
  const request = ++selection;
  creating = true;
  loading = false;
  renderConversation();
  try {
    const thread = await api(apiBase, {method: "POST"});
    rememberThread(thread);
    if (request === selection) {
      active = thread;
      updateUrl(thread, true);
    }
    renderThreadList();
    return request === selection ? thread : null;
  } finally {
    creating = false;
    renderConversation();
  }
}

async function newChat() {
  if (creating) return;
  message("chat-error", "");
  setPanel("conversation");
  try {
    if (await createThread()) {
      question.value = "";
      question.focus();
    }
  } catch (error) {
    message("chat-error", error.message);
  }
}

async function send(text) {
  if (text.length < 3 || question.disabled) return;
  message("chat-error", "");
  let thread = active;
  try {
    thread ??= await createThread();
  } catch (error) {
    message("chat-error", error.message);
    return;
  }
  if (!thread || pending.has(thread.id)) return;
  pending.set(thread.id, text);
  question.value = "";
  renderConversation();
  try {
    const result = await api(apiBase + "/" + thread.id, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: text, revision: thread.revision}),
    });
    rememberThread(result);
    if (active?.id === thread.id) {
      // Another tab load can finish while this turn is in flight.
      if (active.revision === thread.revision) {
        active = {...result, turns: [...active.turns, result.turn]};
      } else if (active.revision < result.revision) {
        await loadThread(thread.id);
      }
    }
  } catch (error) {
    if (active?.id === thread.id) {
      if (error.status === 409) await loadThread(thread.id);
      if (active?.id === thread.id) {
        question.value = text;
        message("chat-error", error.message);
      }
    }
  } finally {
    pending.delete(thread.id);
    renderThreadList();
    renderConversation();
    if (active?.id === thread.id && workspace.dataset.activePanel === "conversation") question.focus();
  }
}

async function initialize() {
  loading = true;
  renderConversation();
  try {
    threads = await api(apiBase);
    const requested = new URLSearchParams(location.search).get("chat");
    const initial = threads.find((thread) => thread.id === requested) ?? threads[0];
    await loadThread(initial?.id);
  } catch (error) {
    loading = false;
    message("threads-status", error.message);
  } finally {
    initializing = false;
    renderConversation();
  }
}

for (const control of document.querySelectorAll("[data-panel]")) {
  control.addEventListener("click", () => setPanel(control.dataset.panel));
}
for (const control of document.querySelectorAll("[data-question]")) {
  control.addEventListener("click", () => send(control.dataset.question));
}
byId("new-chat").addEventListener("click", newChat);
byId("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  send(question.value.trim());
});
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    byId("ask-form").requestSubmit();
  }
});
window.addEventListener("popstate", () => {
  const requested = new URLSearchParams(location.search).get("chat");
  loadThread(requested);
});
byId("page-form").addEventListener("submit", (event) => {
  event.preventDefault();
  showPage(Number(byId("page-number").value));
});
byId("page-number").addEventListener("change", (event) => showPage(Number(event.target.value)));
byId("previous-page").addEventListener("click", () => showPage(page - 1));
byId("next-page").addEventListener("click", () => showPage(page + 1));
byId("pdf-page").addEventListener("error", (event) => {
  event.target.hidden = true;
  byId("page-error").hidden = false;
});

initialize();
