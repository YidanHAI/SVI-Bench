const dimensions = {
  D1: {
    title: "Does the system speak when a target event occurs?",
    description:
      "Match each annotated event with one valid response—without misses, premature fires, or duplicate triggering.",
    anchor: "Exactly one valid trigger",
    count: "55 items",
    color: "#2769be",
    background: "#eef5ff",
  },
  D2: {
    title: "Does the system remain silent when no response is due?",
    description:
      "Measure barge-ins, self-talk, hallucinated events, and long unsolicited monologues over annotated silence intervals.",
    anchor: "No spurious utterance",
    count: "49 items",
    color: "#238875",
    background: "#edf8f4",
  },
  D3: {
    title: "Does a valid response arrive within the required window?",
    description:
      "Measure delay from an annotated visual event or scheduled query to the first token—or to key information when the item defines content latency.",
    anchor: "At most 1 second",
    count: "73 items",
    color: "#df742f",
    background: "#fff4ec",
  },
  D4: {
    title: "Is the response correct and consistent with the scene?",
    description:
      "Check factual accuracy, required information, counting or interval error, response format, and freedom from fabrication.",
    anchor: "Item-specific information coverage",
    count: "75 items",
    color: "#7250a6",
    background: "#f5f0fb",
  },
  D5: {
    title: "Can the system delegate and preserve long-horizon context?",
    description:
      "Test background-agent delegation, faithful recall of earlier visual or dialogue context, and continuity while delegated work runs.",
    anchor: "Correct delegation or faithful recall",
    count: "20 items",
    color: "#167f89",
    background: "#eaf7f7",
  },
};

const results = [
  {
    name: "JoyAI-VL-Interaction",
    overall: 53.61,
    withoutD3: 39.67,
    color: "#3977e8",
  },
  {
    name: "MOSS-VL-Realtime",
    overall: 25.44,
    withoutD3: 26.33,
    color: "#2a9d8f",
  },
  {
    name: "MiniCPM-O-4.5-9B",
    overall: 21.33,
    withoutD3: 20.44,
    color: "#e9a23b",
  },
  {
    name: "Doubao Seed 2.1 Pro",
    overall: 21.22,
    withoutD3: 29.44,
    color: "#b66cac",
  },
  {
    name: "Mage-VL",
    overall: 19.5,
    withoutD3: 21.78,
    color: "#5aa9d6",
  },
];

const header = document.querySelector("[data-header]");
const navToggle = document.querySelector("[data-nav-toggle]");
const navLinks = document.querySelector("[data-nav-links]");

const updateHeader = () => {
  header?.classList.toggle("scrolled", window.scrollY > 18);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

navToggle?.addEventListener("click", () => {
  const open = navToggle.getAttribute("aria-expanded") !== "true";
  navToggle.setAttribute("aria-expanded", String(open));
  navLinks?.classList.toggle("open", open);
  header?.classList.toggle("menu-open", open);
});

navLinks?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    navToggle?.setAttribute("aria-expanded", "false");
    navLinks.classList.remove("open");
    header?.classList.remove("menu-open");
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    navToggle?.setAttribute("aria-expanded", "false");
    navLinks?.classList.remove("open");
    header?.classList.remove("menu-open");
  }
});

const dimensionPanel = document.querySelector("[data-dimension-panel]");
const dimensionNumber = document.querySelector("[data-dimension-number]");
const dimensionTitle = document.querySelector("[data-dimension-title]");
const dimensionDescription = document.querySelector("[data-dimension-description]");
const dimensionAnchor = document.querySelector("[data-dimension-anchor]");
const dimensionCount = document.querySelector("[data-dimension-count]");

function selectDimension(key) {
  const dimension = dimensions[key];
  if (!dimension || !dimensionPanel) return;

  document.querySelectorAll("[data-dimension]").forEach((button) => {
    const selected = button.dataset.dimension === key;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });

  dimensionPanel.style.setProperty("--dimension-color", dimension.color);
  dimensionPanel.style.setProperty("--dimension-bg", dimension.background);
  dimensionPanel.setAttribute("aria-labelledby", `dimension-tab-${key.toLowerCase()}`);
  dimensionNumber.textContent = key;
  dimensionTitle.textContent = dimension.title;
  dimensionDescription.textContent = dimension.description;
  dimensionAnchor.textContent = dimension.anchor;
  dimensionCount.textContent = dimension.count;
}

document.querySelectorAll("[data-dimension]").forEach((button, index, buttons) => {
  button.addEventListener("click", () => selectDimension(button.dataset.dimension));
  button.addEventListener("keydown", (event) => {
    if (!['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft'].includes(event.key)) return;
    event.preventDefault();
    const direction = ['ArrowDown', 'ArrowRight'].includes(event.key) ? 1 : -1;
    const next = (index + direction + buttons.length) % buttons.length;
    buttons[next].focus();
    selectDimension(buttons[next].dataset.dimension);
  });
});

selectDimension("D1");

const chart = document.querySelector("[data-bar-chart]");
const chartEyebrow = document.querySelector("[data-chart-eyebrow]");
const chartTitle = document.querySelector("[data-chart-title]");
const chartDescription = document.querySelector("[data-chart-description]");
const chartNote = document.querySelector("[data-chart-note]");

function renderChart(metric) {
  if (!chart) return;
  const diagnostic = metric === "withoutD3";
  const ordered = [...results].sort((a, b) => b[metric] - a[metric]);
  const scaleMax = 60;

  chart.replaceChildren();
  for (const result of ordered) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.setAttribute("aria-label", `${result.name}: ${result[metric].toFixed(2)}`);

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = result.name;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.setProperty("--bar-color", result.color);
    track.append(fill);

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = result[metric].toFixed(2);

    row.append(label, track, value);
    chart.append(row);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fill.style.width = `${Math.min(100, (result[metric] / scaleMax) * 100)}%`;
      });
    });
  }

  chartEyebrow.textContent = diagnostic ? "Latency sensitivity" : "Official leaderboard";
  chartTitle.textContent = diagnostic ? "Score without D3" : "Overall score";
  chartDescription.textContent = diagnostic
    ? "D1/D2 remain item-coupled before latency is removed"
    : "Item-level D1/D2 coupling · higher is better";
  chartNote.textContent = diagnostic
    ? "Diagnostic only: this view does not replace or rerank the official leaderboard."
    : "Overall is the only leaderboard score. Values are computed before rounding.";
}

document.querySelectorAll("[data-result-metric]").forEach((button) => {
  button.addEventListener("click", () => {
    const metric = button.dataset.resultMetric;
    document.querySelectorAll("[data-result-metric]").forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("active", active);
      candidate.setAttribute("aria-pressed", String(active));
    });
    renderChart(metric);
  });
});

renderChart("overall");

function copyWithFeedback(button, text) {
  if (!button || !text) return;
  navigator.clipboard.writeText(text).then(() => {
    const original = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = original;
    }, 1600);
  }).catch(() => {
    button.textContent = "Select to copy";
  });
}

const copyCode = document.querySelector("[data-copy-code]");
copyCode?.addEventListener("click", () => {
  const source = document.querySelector("[data-command-block]")?.textContent || "";
  const cleaned = source
    .split("\n")
    .map((line) => line.replace(/^\$\s?/, ""))
    .join("\n")
    .trim();
  copyWithFeedback(copyCode, cleaned);
});

const copyCitation = document.querySelector("[data-copy-citation]");
copyCitation?.addEventListener("click", () => {
  const citation = document.querySelector("[data-citation]")?.textContent.trim();
  copyWithFeedback(copyCitation, citation);
});

const revealItems = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.08, rootMargin: "0px 0px -35px" },
  );
  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("visible"));
}
