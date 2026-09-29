import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { indexedDB } from "fake-indexeddb";
import { act, createElement } from "react";

const dom = new JSDOM(
  '<!doctype html><html><body><div id="root"></div></body></html>',
  { url: "https://studio.example.test/" },
);
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.localStorage = dom.window.localStorage;
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.indexedDB = indexedDB;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
dom.window.indexedDB = indexedDB;
sessionStorage.setItem("mirai-access", "app-code");
let input;
const png = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y8KdEAAAAAASUVORK5CYII=",
  "base64",
);
globalThis.fetch = async (url, options) => {
  if (url === "/api/config")
    return new Response(
      JSON.stringify({
        cloudflare: true,
        wai: true,
        configured: true,
        localPreview: false,
      }),
    );
  input = JSON.parse(options.body);
  return new Response(png, {
    headers: {
      "Content-Type": "image/png",
      "X-Image-Seed": String(input.seed),
    },
  });
};

const { createRoot } = await import("react-dom/client");
const { default: App } = await import("../src/App.jsx");
const root = createRoot(document.getElementById("root"));

test("WAI controls send external GPU parameters and the verified eye LoRA trigger, not Cloudflare model names", async () => {
  try {
    await act(async () => {
      root.render(createElement(App));
      await new Promise((resolve) => setImmediate(resolve));
    });
    await act(async () =>
      document
        .querySelector(".provider-switch button:nth-child(2)")
        .dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })),
    );
    assert.match(
      document.querySelector(".generate-button").textContent,
      /Tạo ảnh ngay/,
    );
    await act(async () => {
      document
        .querySelector(".generate-button")
        .dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 70));
    });
    assert.equal(input.provider, "wai");
    assert.equal(input.model, "wai-v17");
    assert.equal(input.steps, 25);
    assert.equal(input.scheduler, "euler_a");
    assert.equal(input.clip_skip, 2);
    assert.deepEqual(input.loras, {
      anatomy: { enabled: true, weight: 0.55 },
      eyes: { enabled: true, weight: 0.45 },
    });
    assert.equal((input.prompt.match(/perfect eyes/gi) || []).length, 1);
    assert.ok(document.querySelector(".result-stage"));
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
  }
});
