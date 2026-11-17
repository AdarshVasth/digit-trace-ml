"use strict";

const element = (id) => document.getElementById(id);
const pixels = Array(64).fill(0);
const pad = element("pad");
const status = element("status");
const result = element("result");
let revision = 0;
let activePointer = null;
let brush = 16;
let lastIndex = null;

function resetResult(message = "Ready to classify.") {
  revision += 1;
  result.hidden = true;
  status.textContent = message;
}

function setPixel(index, value) {
  pixels[index] = value;
  pad.children[index].setAttribute("aria-pressed", String(value > 0));
}

for (let index = 0; index < pixels.length; index += 1) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cell";
  button.dataset.index = index;
  button.setAttribute("aria-pressed", "false");
  button.setAttribute("aria-label", `Row ${Math.floor(index / 8) + 1}, column ${index % 8 + 1}`);
  button.addEventListener("click", (event) => {
    if (event.detail !== 0) return; // Pointer events already paint the cell.
    setPixel(index, pixels[index] ? 0 : 16);
    resetResult();
  });
  pad.append(button);
}

function paintTo(index) {
  if (lastIndex === null) lastIndex = index;
  const x = index % 8;
  const y = Math.floor(index / 8);
  const fromX = lastIndex % 8;
  const fromY = Math.floor(lastIndex / 8);
  const steps = Math.max(Math.abs(x - fromX), Math.abs(y - fromY), 1);
  for (let step = 0; step <= steps; step += 1) {
    const col = Math.round(fromX + (x - fromX) * step / steps);
    const row = Math.round(fromY + (y - fromY) * step / steps);
    setPixel(row * 8 + col, brush);
  }
  lastIndex = index;
}

pad.addEventListener("pointerdown", (event) => {
  const cell = event.target.closest(".cell");
  if (!cell || event.button !== 0 || activePointer !== null) return;
  const index = Number(cell.dataset.index);
  activePointer = event.pointerId;
  brush = pixels[index] ? 0 : 16;
  pad.setPointerCapture(activePointer);
  resetResult();
  paintTo(index);
});

pad.addEventListener("pointermove", (event) => {
  if (event.pointerId !== activePointer) return;
  const cell = document.elementFromPoint(event.clientX, event.clientY)?.closest(".cell");
  if (cell && pad.contains(cell)) paintTo(Number(cell.dataset.index));
  else lastIndex = null;
});

pad.addEventListener("lostpointercapture", () => {
  activePointer = null;
  lastIndex = null;
});

function showNeighbor(neighbor) {
  const figure = document.createElement("figure");
  const preview = document.createElement("canvas");
  preview.width = preview.height = 8;
  preview.setAttribute("aria-hidden", "true");
  const drawing = preview.getContext("2d");
  neighbor.pixels.forEach((value, index) => {
    const level = Math.round(value * 255 / 16);
    drawing.fillStyle = `rgb(${level}, ${level}, ${level})`;
    drawing.fillRect(index % 8, Math.floor(index / 8), 1, 1);
  });
  const caption = document.createElement("figcaption");
  caption.textContent = `Digit ${neighbor.label} · ${neighbor.distance.toFixed(1)} away`;
  figure.append(preview, caption);
  return figure;
}

async function classify() {
  resetResult("Classifying…");
  const current = revision;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({pixels}),
    });
    const data = await response.json();
    if (current !== revision) return;
    if (!response.ok) throw new Error(data.error);
    element("digit").textContent = data.digit;
    element("votes").textContent = Object.entries(data.votes)
      .map(([digit, count]) => `${digit}: ${count}/5 votes`).join(" · ");
    element("neighbors").replaceChildren(...data.neighbors.map(showNeighbor));
    status.textContent = `Predicted digit: ${data.digit}.`;
    result.hidden = false;
  } catch (error) {
    if (current === revision) status.textContent = error.message;
  }
}

element("predict").addEventListener("click", classify);
element("clear").addEventListener("click", () => {
  pixels.forEach((_, index) => setPixel(index, 0));
  resetResult("Grid cleared.");
});
element("sample").addEventListener("click", async () => {
  resetResult("Loading sample…");
  const current = revision;
  try {
    const response = await fetch("/sample.json");
    if (!response.ok) throw new Error("Could not load the sample.");
    const sample = await response.json();
    if (current !== revision) return;
    sample.pixels.forEach((value, index) => setPixel(index, value));
    await classify();
  } catch (error) {
    if (current === revision) status.textContent = error.message;
  }
});

async function showMetrics() {
  const response = await fetch("/api/metrics");
  if (!response.ok) throw new Error();
  const metrics = await response.json();
  element("metrics").textContent =
    `Held-out dataset accuracy: ${(metrics.accuracy * 100).toFixed(1)}% on ${metrics.test_count} images.`;
}
showMetrics().catch(() => { element("metrics").textContent = "Could not load evaluation results."; });
