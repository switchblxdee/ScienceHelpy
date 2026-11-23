(() => {
  "use strict";

  const storageKey = "sciencehelpy-theme";
  const system = matchMedia("(prefers-color-scheme: dark)");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  let changing = false;
  const root = document.documentElement;
  let selection;
  try {
    selection = localStorage.getItem(storageKey);
  } catch {
    // The theme still works when storage is unavailable.
  }
  const chosen = (value) => value === "dark" || value === "light";
  const systemTheme = () => system.matches ? "dark" : "light";

  function apply() {
    root.dataset.theme = chosen(selection) ? selection : systemTheme();
    const dark = root.dataset.theme === "dark";
    for (const button of document.querySelectorAll("[data-theme-toggle]")) {
      button.setAttribute("aria-pressed", String(dark));
      button.title = dark ? tr("Включить светлую тему") : tr("Включить тёмную тему");
    }
  }

  apply();
  system.addEventListener("change", apply);
  addEventListener("storage", (event) => {
    if (event.key === storageKey || event.key === null) {
      selection = event.newValue;
      apply();
    }
  });
  addEventListener("DOMContentLoaded", () => {
    apply();
    for (const button of document.querySelectorAll("[data-theme-toggle]")) {
      button.addEventListener("click", () => {
        if (changing) return;
        const next = root.dataset.theme === "dark" ? "light" : "dark";
        const commit = () => {
          selection = next;
          try {
            localStorage.setItem(storageKey, selection);
          } catch {
            // Keep the current tab's selection.
          }
          apply();
        };
        if (!document.startViewTransition || reduced.matches) {
          commit();
          return;
        }
        changing = true;
        root.classList.add("theme-switching");
        const transition = document.startViewTransition(commit);
        transition.finished.catch(() => {}).finally(() => {
          changing = false;
          root.classList.remove("theme-switching");
        });
      });
    }
  });
})();
