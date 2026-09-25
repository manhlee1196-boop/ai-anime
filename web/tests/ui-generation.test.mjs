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
sessionStorage.setItem("mirai-access", "my-app-access-code");
const png = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y8KdEAAAAAASUVORK5CYII=",
  "base64",
);
const calls = [];
globalThis.fetch = async (url, options) => {
  if (url === "/api/config")
    return new Response(
      JSON.stringify({
        cloudflare: true,
        wai: false,
        configured: true,
        localPreview: false,
      }),
      { headers: { "Content-Type": "application/json" } },
    );
  if (url !== "/api/generate") throw Error(`Unexpected request: ${url}`);
  calls.push({ url, options });
  const payload = JSON.parse(options.body);
  return new Response(png, {
    headers: {
      "Content-Type": "image/png",
      "X-Image-Seed": String(payload.seed),
      "X-Generation-Ms": "1234",
    },
  });
};

const { createRoot } = await import("react-dom/client");
const { default: App } = await import("../src/App.jsx");
const { listGenerations } = await import("../src/lib/history.js");
const root = createRoot(document.getElementById("root"));

test("configured studio sends real generation requests, saves the returned image, and exposes download/history", async () => {
  try {
    await act(async () => {
      root.render(createElement(App));
      await new Promise((resolve) => setImmediate(resolve));
    });
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
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "/api/generate");
    assert.equal(
      calls[0].options.headers.Authorization,
      "Bearer my-app-access-code",
    );
    const sent = JSON.parse(calls[0].options.body);
    assert.equal(sent.provider, "cloudflare");
    assert.equal(sent.model, "sdxl-base");
    assert.equal(sent.mode, "txt2img");
    assert.equal(sent.steps, 20);
    assert.equal(typeof sent.seed, "number");
    assert.equal(
      "loras" in sent,
      false,
      "Workers AI must not receive WAI-only settings",
    );

    const records = await listGenerations();
    assert.equal(records.length, 1);
    assert.equal(records[0].blob.type, "image/png");
    assert.equal(records[0].seed, sent.seed);
    assert.ok(
      document.querySelector(".result-stage"),
      "new result should replace sample hero",
    );
    assert.equal(document.querySelectorAll(".gallery-card").length, 1);
    assert.match(
      document.querySelector(".result-actions").textContent,
      /Đã lưu trong trình duyệt/,
    );
    assert.ok(
      document.querySelector('.result-actions button[title="Tải ảnh về"]'),
    );
  } finally {
    await act(async () => root.unmount());
    dom.window.close();
  }
});
