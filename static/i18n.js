"use strict";

const uiLanguage = document.documentElement.lang;
const uiStrings = JSON.parse(document.getElementById("ui-strings").textContent);
function tr(text, values = {}) {
  const translated = Object.hasOwn(uiStrings, text) ? uiStrings[text] : text;
  return translated.replace(/\{([a-z]+)\}/g, (placeholder, key) =>
    Object.hasOwn(values, key) ? String(values[key]) : placeholder);
}
