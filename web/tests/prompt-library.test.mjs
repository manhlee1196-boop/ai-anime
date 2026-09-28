import test from "node:test";
import assert from "node:assert/strict";
import {
  clearPersistedLibrary,
  filterLibraryItems,
  libraryToText,
  parsePromptLibrary,
  persistLibrary,
  readPersistedLibrary,
  stripDiacritics,
} from "../src/lib/promptLibrary.js";
import { SAMPLE_LIBRARY_TEXT } from "../src/lib/sampleLibrary.js";

const TEXT_FORMAT = `=== 50 PROMPTS – ÁO SƠ MI VĂN PHÒNG ===

Tên tiếng Việt | Nội dung prompts tiếng Anh

Nhân vật công sở, trang phục lịch sự, ánh sáng trong trẻo

PROMPT 01 - Nữ thư ký ngồi bàn làm việc mỉm cười
Masterpiece, best quality, ultra-detailed anime style, side view,
office worker sitting at a tidy desk, white shirt, glasses,
warm desk lamp light, modern office at night, high detail.

PROMPT 02 - Nữ thư ký đứng bên cửa sổ
Prompt: Masterpiece, best quality, ultra-detailed anime style, front view,
office worker standing by a large office window, city night view.
Negative: low quality, blurry, extra fingers
Steps: 28
CFG: 7.5
Size: 832 x 1216
Seed: 424242

03 - Chuyến tàu cuối ngày
Masterpiece, best quality, office worker on a night train, window reflections.
`;

test("parsePromptLibrary reads the PROMPT NN - Tên tiếng Việt file format", () => {
  const library = parsePromptLibrary(TEXT_FORMAT, { name: "ao-so-mi.txt" });
  assert.equal(library.name, "50 PROMPTS – ÁO SƠ MI VĂN PHÒNG");
  assert.equal(library.items.length, 3);
  assert.equal(library.items[0].index, 1);
  assert.equal(library.items[0].title, "Nữ thư ký ngồi bàn làm việc mỉm cười");
  assert.match(library.items[0].prompt, /^Masterpiece, best quality/);
  assert.match(
    library.items[0].prompt,
    /modern office at night, high detail\.$/,
  );
  assert.ok(
    !library.items[0].prompt.includes("\n"),
    "multi-line prompt bodies should be joined into one line",
  );
  assert.match(library.description, /Nhân vật công sở/);

  const tuned = library.items[1];
  assert.equal(tuned.title, "Nữ thư ký đứng bên cửa sổ");
  assert.equal(tuned.negative, "low quality, blurry, extra fingers");
  assert.equal(tuned.steps, 28);
  assert.equal(tuned.cfg, 7.5);
  assert.equal(tuned.width, 832);
  assert.equal(tuned.height, 1216);
  assert.equal(tuned.seed, 424242);
  assert.match(
    tuned.prompt,
    /^Masterpiece, best quality, ultra-detailed anime style, front view/,
    'the "Prompt:" prefix is content, not a heading, and gets stripped',
  );
  assert.ok(
    !tuned.prompt.includes("Negative:"),
    "directive lines must not leak into the prompt",
  );

  // Tiêu đề dạng "03 - Tên" (không có chữ PROMPT) vẫn được nhận khi số khớp.
  assert.equal(library.items[2].index, 3);
  assert.equal(library.items[2].title, "Chuyến tàu cuối ngày");
});

test("parsePromptLibrary reads one-line table rows, JSON and plain paragraphs", () => {
  const table = parsePromptLibrary(
    [
      "=== 3 PROMPTS – BẢNG ===",
      "Tên tiếng Việt | Nội dung prompts tiếng Anh",
      "Cô gái bên hồ | masterpiece, best quality, anime girl reading by a lake at dawn",
      "Phố mưa neon | masterpiece, best quality, rainy neon street, reflective asphalt",
      "Vườn trên mây | masterpiece, best quality, floating garden above the clouds",
    ].join("\n"),
  );
  assert.equal(table.items.length, 3);
  assert.equal(table.items[0].title, "Cô gái bên hồ");
  assert.match(table.items[2].prompt, /floating garden above the clouds/);

  const json = parsePromptLibrary(
    JSON.stringify([
      { title: "Chân dung", prompt: "masterpiece, anime portrait, soft light" },
      { name: "Phong cảnh", en: "masterpiece, anime landscape, wide sky" },
    ]),
  );
  assert.equal(json.items.length, 2);
  assert.equal(json.items[1].title, "Phong cảnh");

  const plain = parsePromptLibrary(
    "masterpiece, anime girl under cherry blossoms\n\nmasterpiece, anime boy on a rooftop at sunset",
  );
  assert.equal(plain.items.length, 2);
  assert.match(plain.items[1].prompt, /rooftop at sunset/);
});

test("parsePromptLibrary rejects empty and unparseable files", () => {
  assert.throws(() => parsePromptLibrary("   "), /File trống/);
  assert.throws(
    () => parsePromptLibrary("Tên tiếng Việt | Nội dung prompts tiếng Anh"),
    /Không tìm thấy prompt nào/,
  );
});

test("prompt bodies longer than the studio limit are trimmed and flagged", () => {
  const long = parsePromptLibrary(
    `PROMPT 01 - Prompt rất dài\n${"masterpiece, ".repeat(400)}`,
  );
  assert.equal(long.items[0].prompt.length, 2000);
  assert.equal(long.items[0].truncated, true);
  assert.equal(long.truncated, 1);
});

test("filterLibraryItems matches Vietnamese titles without diacritics", () => {
  const { items } = parsePromptLibrary(TEXT_FORMAT);
  assert.equal(stripDiacritics("Áo sơ mi"), "Ao so mi");
  assert.equal(filterLibraryItems(items, "thu ky").length, 2);
  assert.equal(filterLibraryItems(items, "cửa sổ").length, 1);
  assert.equal(filterLibraryItems(items, "  ").length, items.length);
  assert.equal(filterLibraryItems(items, "khong co ket qua").length, 0);
});

test("libraryToText round-trips through parsePromptLibrary", () => {
  const original = parsePromptLibrary(SAMPLE_LIBRARY_TEXT, {
    name: "Thư viện mẫu",
  });
  assert.equal(original.items.length, 12);
  const exported = libraryToText(original);
  // Tên thư viện đã bắt đầu bằng "12 PROMPTS –" nên không bị lặp hai lần.
  assert.match(exported, /^=== 12 PROMPTS – NHÂN VẬT VĂN PHÒNG/);
  const again = parsePromptLibrary(exported, {
    name: "round-trip",
  });
  assert.equal(again.items.length, original.items.length);
  assert.deepEqual(
    again.items.map((item) => item.prompt),
    original.items.map((item) => item.prompt),
  );
  assert.equal(again.items[7].negative, original.items[7].negative);
  assert.equal(again.items[3].steps, 26);
  assert.equal(again.items[3].width, 1216);

  // Tên thư viện đã sẵn dạng "50 PROMPTS – CHỦ ĐỀ" thì không bị lặp lại.
  const numbered = parsePromptLibrary(TEXT_FORMAT, { name: "50-prompts.txt" });
  assert.match(
    libraryToText(numbered),
    /^=== 3 PROMPTS – ÁO SƠ MI VĂN PHÒNG ===/,
  );
});

test("prompt library persists in localStorage between sessions", () => {
  globalThis.localStorage = {
    store: new Map(),
    getItem(key) {
      return this.store.has(key) ? this.store.get(key) : null;
    },
    setItem(key, value) {
      this.store.set(key, String(value));
    },
    removeItem(key) {
      this.store.delete(key);
    },
  };
  const library = parsePromptLibrary(SAMPLE_LIBRARY_TEXT, {
    name: "Thư viện mẫu",
  });
  assert.equal(persistLibrary(library), true);
  const restored = readPersistedLibrary();
  assert.equal(restored.items.length, 12);
  assert.equal(restored.name, library.name);
  assert.match(restored.name, /NHÂN VẬT VĂN PHÒNG/);
  clearPersistedLibrary();
  assert.equal(readPersistedLibrary(), null);
  assert.equal(persistLibrary({ name: "x", items: [] }), true);
  assert.equal(readPersistedLibrary(), null);
});
