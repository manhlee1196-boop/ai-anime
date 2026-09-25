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
const root = createRoot(document.getElementById("root"));

function click(selector) {
  const button = document.querySelector(selector);
  assert.ok(button, `Missing clickable element ${selector}`);
  button.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
}

test("studio renders, responds to model/mode controls, and never simulates generation in Vite preview", async () => {
  try {
    await act(async () => {
      root.render(createElement(App));
      await new Promise((resolve) => setImmediate(resolve));
    });
    assert.match(document.body.textContent, /bản xem trước giao diện/i);
    assert.match(document.body.textContent, /ẢNH MINH HỌA/);
    assert.equal(document.querySelectorAll(".inspiration-card").length, 3);
    assert.equal(document.querySelector(".lora-section"), null);

    await act(async () => click(".provider-switch button:nth-child(2)"));
    assert.ok(
      document.querySelector(".lora-section"),
      "WAI-specific LoRA settings should show only for WAI",
    );
    assert.match(
      document.querySelector(".lora-section").textContent,
      /Worker không tự tải/i,
    );

    await act(async () => click(".provider-switch button:first-child"));
    assert.equal(document.querySelector(".lora-section"), null);
    await act(async () => click(".mode-select button:nth-child(3)"));
    assert.ok(
      document.querySelector(".upload-area"),
      "Inpainting mode should show a source image input",
    );

    await act(async () => click(".generate-button"));
    assert.match(
      document.querySelector(".toast").textContent,
      /bản xem trước/i,
    );
    assert.equal(
      generationCalls,
      0,
      "Vite preview must not call the inference endpoint",
    );

    await act(async () => click(".top-nav button"));
    assert.ok(document.querySelector(".help-dialog"));
    await act(async () => click(".help-dialog .dialog-close"));
    assert.equal(document.querySelector(".help-dialog"), null);
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
  }
});
