(() => {
  "use strict";

  const canvas = document.querySelector(".hero-wave");
  const ink = canvas?.getContext("2d");
  if (!ink) return;

  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const duration = 3650;
  let width = 0;
  let height = 0;
  let frame = 0;
  let started;
  let progress = 0;

  function point(u, v, time) {
    const left = u + 0.33;
    const right = u - 0.36;
    const depth = v - 0.45;
    const envelope = Math.exp(-2.8 * (u * u + depth * depth));
    const ripple = Math.sin(11 * Math.hypot(left, depth) - time)
      + 0.65 * Math.sin(12 * Math.hypot(right, depth) - time + 1.5);
    return [
      width * (0.54 + u * (0.23 + 0.27 * v)),
      height * (0.19 + 0.67 * v - ripple * envelope * (0.045 + 0.042 * v)),
    ];
  }

  function trace(sample, steps, color, thickness) {
    ink.beginPath();
    for (let i = 0; i <= steps; i++) {
      const [x, y] = sample(i / steps);
      if (i === 0) ink.moveTo(x, y);
      else ink.lineTo(x, y);
    }
    ink.strokeStyle = color;
    ink.lineWidth = thickness;
    ink.stroke();
  }

  function draw(value) {
    if (!width || !height) return;
    ink.clearRect(0, 0, width, height);
    const phase = 2.6 * (1 - Math.pow(1 - value, 3));
    for (let i = 0; i <= 22; i++) {
      const v = i / 22;
      const highlighted = i === 11 || i === 16;
      trace(t => point(2 * t - 1, v, phase), 64,
        highlighted ? "#ff7169" : `rgba(245,101,95,${0.22 + 0.36 * v})`, highlighted ? 1.7 : 1);
    }
    for (let i = 0; i <= 24; i++) {
      const u = (i / 24) * 2 - 1;
      trace(t => point(u, t, phase), 32, "rgba(245,165,153,0.19)", 1);
    }
    for (const [u, v] of [[-0.53, 0.5], [0.09, 16 / 22], [0.62, 0.5]]) {
      const [x, y] = point(u, v, phase);
      ink.beginPath();
      ink.arc(x, y, 6, 0, Math.PI * 2);
      ink.fillStyle = "#ff6c6240";
      ink.fill();
      ink.beginPath();
      ink.arc(x, y, 2.6, 0, Math.PI * 2);
      ink.fillStyle = "#ffc5b9";
      ink.fill();
    }
  }

  function tick(time) {
    started ??= time;
    progress = Math.min((time - started) / duration, 1);
    draw(progress);
    if (progress < 1) frame = requestAnimationFrame(tick);
  }

  function motionChanged() {
    cancelAnimationFrame(frame);
    if (reduced.matches) {
      progress = 1;
      draw(progress);
    } else if (progress < 1) {
      draw(progress);
      frame = requestAnimationFrame(tick);
    }
  }

  new ResizeObserver(([entry]) => {
    ({width, height} = entry.contentRect);
    if (!width || !height) return;
    const density = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * density);
    canvas.height = Math.round(height * density);
    ink.setTransform(density, 0, 0, density, 0, 0);
    draw(reduced.matches ? 1 : progress);
  }).observe(canvas);
  reduced.addEventListener("change", motionChanged);
  motionChanged();
})();
