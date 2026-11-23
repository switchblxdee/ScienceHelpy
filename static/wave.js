(() => {
  "use strict";

  const canvas = document.querySelector(".archive-wave");
  const ink = canvas?.getContext("2d");
  if (!ink) return;

  const root = document.documentElement;
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const width = canvas.width;
  const height = canvas.height;
  let frame = 0;
  let started;
  let progress = motion.matches ? 1 : 0;

  function draw() {
    const palette = getComputedStyle(root);
    ink.clearRect(0, 0, width, height);
    ink.fillStyle = palette.getPropertyValue("--plot-grid").trim();
    for (let x = 8; x < width; x += 8) {
      for (let y = 4; y < height; y += 8) ink.fillRect(x, y, 1, 1);
    }
    ink.fillStyle = palette.getPropertyValue("--line").trim();
    ink.fillRect(0, 44, width, 1);
    ink.fillRect(82, 0, 1, height);

    const signal = palette.getPropertyValue("--accent").trim();
    const echo = palette.getPropertyValue("--accent-line").trim();
    for (let x = 4; x < 4 + Math.floor((width - 8) * progress); x += 2) {
      const distance = (x - 83) / 36;
      const ripple = 29 * Math.exp(-(distance ** 2));
      ink.fillStyle = echo;
      ink.fillRect(x, Math.round(44 - ripple), 2, 1);
      ink.fillRect(x, Math.round(44 + ripple), 2, 1);
      ink.fillStyle = signal;
      ink.fillRect(x, Math.round(44 + ripple * Math.sin((x - 83) * 0.29)), 2, 2);
    }
  }

  function step(time) {
    started ??= time;
    progress = Math.min(1, (time - started) / 700);
    draw();
    if (progress < 1) frame = requestAnimationFrame(step);
  }

  function stopAnimation() {
    if (!motion.matches) return;
    cancelAnimationFrame(frame);
    progress = 1;
    draw();
  }

  new MutationObserver(draw).observe(root, {attributes: true, attributeFilter: ["data-theme"]});
  motion.addEventListener("change", stopAnimation);
  draw();
  if (!motion.matches) frame = requestAnimationFrame(step);
})();
