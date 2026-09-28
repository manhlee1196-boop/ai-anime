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
dom.window.Element.prototype.scrollIntoView = () => {};

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
  return new Response(JSON.stringify({ error: "Preview does not infer" }), {
    status: 503,
  });
};

const { createRoot } = await import("react-dom/client");
const { default: App } = await import("../src/App.jsx");

const PROMPT_FIELD = '[aria-label="Prompt tạo ảnh"]';
const NEGATIVE_FIELD = '[aria-label="Negative prompt"]';

function click(element) {
  assert.ok(element, "Missing element to click");
  element.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
}

function clickByText(selector, label) {
  const found = [...document.querySelectorAll(selector)].find((node) =>
    node.textContent.includes(label),
  );
  assert.ok(found, `Missing "${label}" among ${selector}`);
  click(found);
}

function type(selector, value) {
  const element = document.querySelector(selector);
  assert.ok(element, `Missing input ${selector}`);
  const prototype =
    element instanceof dom.window.HTMLTextAreaElement
      ? dom.window.HTMLTextAreaElement.prototype
      : dom.window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value").set.call(element, value);
  element.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
}

async function mount() {
  const root = createRoot(document.getElementById("root"));
  await act(async () => {
    root.render(createElement(App));
    await new Promise((resolve) => setImmediate(resolve));
  });
  return root;
}

function promptValue() {
  return document.querySelector(PROMPT_FIELD).value;
}

function rows() {
  return [...document.querySelectorAll(".library-item")];
}

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setImmediate(resolve));
  });
}

test("prompt library loads a .txt file and fills the prompt field on selection", async () => {
  const root = await mount();
  try {
    // Chưa nạp thư viện: nút "Thư viện" bị khóa, thanh nạp file vẫn hiện.
    assert.ok(document.querySelector(".library-bar"));
    assert.equal(
      document.querySelector(".prompt-actions .subtle-action").disabled,
      true,
    );
    assert.equal(document.querySelector(".library-dialog"), null);

    const fileText = [
      "=== 2 PROMPTS – THƯ VIỆN KIỂM THỬ ===",
      "",
      "Tên tiếng Việt | Nội dung prompts tiếng Anh",
      "",
      "PROMPT 01 - Cô gái dưới hoa anh đào",
      "masterpiece, best quality, anime girl under cherry blossoms, soft dusk light",
      "Negative: low quality, blurry, extra fingers",
      "Steps: 18",
      "",
      "PROMPT 02 - Nam thanh niên trên sân thượng",
      "masterpiece, best quality, anime boy on a rooftop at night, neon city",
      "Steps: 30",
      "CFG: 8",
    ].join("\n");
    const file = new dom.window.File([fileText], "50-prompts.txt", {
      type: "text/plain",
    });
    const input = document.querySelector(
      'input[aria-label="Nạp file danh sách prompt"]',
    );
    assert.ok(input, "Hidden prompt-file input must exist");
    Object.defineProperty(input, "files", {
      value: [file],
      configurable: true,
    });
    await act(async () => {
      input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
      await new Promise((resolve) => setImmediate(resolve));
    });

    // Danh sách prompt hiện ra, đúng số dòng và đúng tên file làm tiêu đề.
    const dialog = document.querySelector(".library-dialog");
    assert.ok(dialog, "Library dialog should open after import");
    assert.match(dialog.querySelector("h2").textContent, /THƯ VIỆN KIỂM THỬ/);
    assert.equal(rows().length, 2);
    assert.match(rows()[0].textContent, /Cô gái dưới hoa anh đào/);
    assert.match(
      document.querySelector(".toast").textContent,
      /Đã nạp 2 prompt từ 2 PROMPTS – THƯ VIỆN KIỂM THỬ/,
    );
    assert.equal(
      JSON.parse(localStorage.getItem("mirai-prompt-library-v1")).items.length,
      2,
    );

    // Tìm kiếm bỏ dấu tiếng Việt.
    type('[aria-label="Tìm prompt trong thư viện"]', "san thuong");
    await settle();
    assert.equal(rows().length, 1);
    assert.match(rows()[0].textContent, /Nam thanh niên trên sân thượng/);

    // Chọn prompt → hệ thống tự nạp vào ô Prompt kèm Steps trong file.
    await act(async () => {
      click(rows()[0].querySelector(".library-item-main"));
      await new Promise((resolve) => setImmediate(resolve));
    });
    assert.equal(document.querySelector(".library-dialog"), null);
    assert.equal(
      promptValue(),
      "masterpiece, best quality, anime boy on a rooftop at night, neon city",
    );
    // Steps: 30 trong file vượt dải trượt của SDXL (tối đa 20) nên bị kẹp lại.
    assert.equal(
      document.querySelector('[aria-label="Số bước (steps)"]').value,
      "20",
    );
    assert.equal(
      document.querySelector('[aria-label="CFG / Guidance"]').value,
      "8",
    );
    assert.match(document.querySelector(".toast").textContent, /Đã nạp prompt/);

    // Thư viện vẫn còn đó, mở lại bằng nút "Thư viện" và chọn prompt thứ nhất.
    await act(async () => {
      clickByText(".prompt-actions .subtle-action", "Thư viện");
      await new Promise((resolve) => setImmediate(resolve));
    });
    assert.equal(rows().length, 2);
    await act(async () => {
      click(rows()[0].querySelector(".library-item-main"));
      await new Promise((resolve) => setImmediate(resolve));
    });
    assert.equal(
      promptValue(),
      "masterpiece, best quality, anime girl under cherry blossoms, soft dusk light",
    );
    assert.equal(
      document.querySelector(NEGATIVE_FIELD).value,
      "low quality, blurry, extra fingers",
    );
    assert.equal(
      document.querySelector('[aria-label="Số bước (steps)"]').value,
      "18",
    );
  } finally {
    root.unmount();
  }
});

test("prompt library supports the bundled sample, append mode, and removal", async () => {
  const root = await mount();
  try {
    // Thư viện đã lưu ở phiên trước được khôi phục.
    const restored = document.querySelector(".library-status b");
    assert.ok(restored, "Stored library should be restored on load");
    assert.match(restored.textContent, /THƯ VIỆN KIỂM THỬ/);

    await act(async () => {
      clickByText(".library-actions .chip-action", "Thư viện mẫu");
      await new Promise((resolve) => setImmediate(resolve));
    });
    assert.equal(rows().length, 12);
    await act(async () => {
      click(rows()[0].querySelector(".library-item-main"));
      await new Promise((resolve) => setImmediate(resolve));
    });
    const samplePrompt = promptValue();
    assert.match(samplePrompt, /^Masterpiece, best quality, ultra-detailed/);

    // Bật "Nối vào prompt hiện tại" rồi chọn tiếp một prompt khác.
    await act(async () => {
      clickByText(".prompt-actions .subtle-action", "Thư viện");
      await new Promise((resolve) => setImmediate(resolve));
    });
    const checkbox = document.querySelector(
      '[aria-label="Nối prompt vào cuối prompt hiện tại"]',
    );
    await act(async () => {
      Object.getOwnPropertyDescriptor(
        dom.window.HTMLInputElement.prototype,
        "checked",
      ).set.call(checkbox, true);
      checkbox.dispatchEvent(new dom.window.Event("click", { bubbles: true }));
      await new Promise((resolve) => setImmediate(resolve));
    });
    await act(async () => {
      click(rows()[1].querySelector(".library-item-main"));
      await new Promise((resolve) => setImmediate(resolve));
    });
    const appended = promptValue();
    assert.ok(appended.startsWith(samplePrompt));
    assert.match(appended, /, Masterpiece, best quality, ultra-detailed/);

    // Gỡ thư viện.
    await act(async () => {
      click(document.querySelector(".library-actions .chip-action.danger"));
      await new Promise((resolve) => setImmediate(resolve));
    });
    assert.equal(document.querySelector(".library-status b"), null);
    assert.equal(localStorage.getItem("mirai-prompt-library-v1"), null);
    assert.match(
      document.querySelector(".toast").textContent,
      /Đã gỡ thư viện prompt/,
    );
  } finally {
    root.unmount();
  }
});

test("prompt library reports a clear error for files without prompts", async () => {
  const root = await mount();
  try {
    const input = document.querySelector(
      'input[aria-label="Nạp file danh sách prompt"]',
    );
    Object.defineProperty(input, "files", {
      value: [
        new dom.window.File(
          ["Tên tiếng Việt | Nội dung prompts tiếng Anh"],
          "rong.txt",
          { type: "text/plain" },
        ),
      ],
      configurable: true,
    });
    await act(async () => {
      input.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
      await new Promise((resolve) => setImmediate(resolve));
    });
    assert.equal(document.querySelector(".library-dialog"), null);
    assert.match(
      document.querySelector(".toast").textContent,
      /Không tìm thấy prompt nào/,
    );
  } finally {
    root.unmount();
  }
});
