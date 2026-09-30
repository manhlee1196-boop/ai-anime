import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { indexedDB } from "fake-indexeddb";
import { act, createElement } from "react";

const dom = new JSDOM(
  '<!doctype html><html><body><div id="root"></div></body></html>',
  { url: "http://localhost:5173/" },
);
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.localStorage = dom.window.localStorage;
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.indexedDB = indexedDB;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
dom.window.indexedDB = indexedDB;

let generationCalls = 0;
globalThis.fetch = async (url) => {
  if (url === "/api/config")
    return new Response(
      JSON.stringify({
        cloudflare: false,
        wai: false,
        configured: false,
        localPreview: true,
      }),
      { headers: { "Content-Type": "application/json" } },
    );
  generationCalls++;
  return new Response(JSON.stringify({ error: "Preview does not infer" }), {
    status: 503,
  });
};

const { createRoot } = await import("react-dom/client");
const { default: App } = await import("../src/App.jsx");
const { NEGATIVE_PRESETS, PROMPT_ORDER } =
  await import("../src/lib/presets.js");
const root = createRoot(document.getElementById("root"));

function click(element) {
  assert.ok(element, "Missing clickable element");
  element.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
}

test("negative presets stay short, unique and load into the visible field", async () => {
  const ids = NEGATIVE_PRESETS.map((preset) => preset.id);
  assert.equal(new Set(ids).size, ids.length, "preset id must be unique");
  for (const preset of NEGATIVE_PRESETS) {
    assert.ok(preset.name.trim(), `${preset.id} needs a label`);
    assert.ok(preset.hint.trim(), `${preset.id} needs an explanation`);
    // Ô negative của giao diện giới hạn 1500 ký tự; bộ dài nhất vẫn ngắn hơn nhiều.
    assert.ok(preset.tags.length <= 500, `${preset.id} is too long`);
    assert.ok(
      !preset.tags.startsWith(",") && !preset.tags.endsWith(","),
      `${preset.id} has stray commas`,
    );
    assert.ok(!preset.tags.includes("  "), `${preset.id} has double spaces`);
    const tags = preset.tags.split(",");
    assert.equal(new Set(tags).size, tags.length, `${preset.id} repeats a tag`);
  }
  assert.ok(PROMPT_ORDER.length >= 8, "prompt order guide is incomplete");
  assert.equal(PROMPT_ORDER[0], "chất lượng");
  assert.equal(PROMPT_ORDER.at(-1), "absurdres");

  try {
    await act(async () => {
      root.render(createElement(App));
      await new Promise((resolve) => setImmediate(resolve));
    });

    assert.match(
      document.body.textContent,
      /Viết thẻ theo thứ tự CLIP ưu tiên/,
      "prompt order guide should be visible next to the prompt box",
    );

    // Mục "Tuỳ chọn nâng cao" mở sẵn nên ô negative và các bộ negative hiện ngay.
    const chips = document.querySelectorAll(
      ".advanced-content .style-chips button",
    );
    assert.equal(chips.length, NEGATIVE_PRESETS.length);

    const box = document.querySelector(
      'textarea[aria-label="Negative prompt"]',
    );
    const before = box.value;
    const core = NEGATIVE_PRESETS.find((preset) => preset.id === "core");
    const chip = [...chips].find((item) =>
      item.textContent.includes(core.name),
    );
    await act(async () => click(chip));
    assert.equal(
      document.querySelector('textarea[aria-label="Negative prompt"]').value,
      core.tags,
      "the preset must be written into the visible negative field",
    );
    assert.notEqual(before, core.tags);
    assert.match(
      document.querySelector(".toast").textContent,
      /Đã nạp negative Illustrious chuẩn/,
    );
    assert.equal(
      generationCalls,
      0,
      "loading a negative preset must not call the inference endpoint",
    );
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
  }
});
