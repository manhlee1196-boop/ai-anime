import test from "node:test";
import assert from "node:assert/strict";
import { indexedDB } from "fake-indexeddb";
import {
  deleteGeneration,
  listGenerations,
  saveGeneration,
  setFavorite,
} from "../src/lib/history.js";

globalThis.window = { indexedDB };
globalThis.indexedDB = indexedDB;

const image = new Blob(["small image"], { type: "image/png" });
const record = (n, favorite = false) => ({
  id: `image-${n}`,
  createdAt: n,
  favorite,
  blob: image,
  prompt: `prompt ${n}`,
});

test("IndexedDB retains recent results, keeps favorites, and never persists transient object URLs", async () => {
  for (let i = 1; i <= 35; i++)
    await saveGeneration(record(i, i === 1 || i === 2));
  let items = await listGenerations();
  assert.equal(items.length, 22); // 20 latest non-favorites plus two older favorites
  assert.equal(items[0].id, "image-35");
  assert.equal(items.at(-1).id, "image-1");
  assert.ok(items.some((item) => item.id === "image-2"));
  assert.ok(!items.some((item) => item.id === "image-3"));

  const first = items.find((item) => item.id === "image-35");
  await setFavorite({ ...first, url: "blob:should-not-persist" }, true);
  items = await listGenerations();
  assert.equal(items.find((item) => item.id === "image-35").favorite, true);
  assert.equal("url" in items.find((item) => item.id === "image-35"), false);
  await deleteGeneration("image-35");
  assert.ok(!(await listGenerations()).some((item) => item.id === "image-35"));
});
