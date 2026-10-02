import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  parseTags,
  filterTags,
  appendTag,
  CATEGORIES,
} from "../src/lib/tags.js";

test("CSV parsing, alias search, category/theme filtering and sorting", () => {
  const tags = parseTags(
    'name,category,count,aliases\r\nlong_hair,0,100,"longhair,\"\"hair\"\""\r\nblue_eyes,0,50,',
  );
  assert.equal(tags.length, 2);
  assert.equal(tags[0].aliases, 'longhair,"hair"');
  assert.equal(filterTags(tags, { query: "longhair" })[0].name, "long_hair");
  assert.equal(filterTags(tags, { query: "LONG HAIR" }).length, 1);
  assert.equal(filterTags(tags, { category: "1" }).length, 0);
  assert.equal(filterTags(tags, { theme: "appearance" }).length, 2);
  assert.equal(filterTags(tags, { sort: "name" })[0].name, "blue_eyes");
});
test("append prevents duplicates and enforces limits without truncation", () => {
  assert.equal(appendTag("", "solo", 4).value, "solo");
  assert.equal(appendTag("scene, ", "solo", 20).value, "scene, solo");
  assert.ok(appendTag("Long Hair, solo", "long_hair", 2000).error);
  assert.ok(appendTag("scene", "solo", 10).error);
  assert.equal(appendTag("solo_focus", "solo", 100).value, "solo_focus, solo");
});
test("real dataset parses completely with recognized categories", async () => {
  const text = await readFile(
    new URL(
      "../../danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv",
      import.meta.url,
    ),
    "utf8",
  );
  const tags = parseTags(text);
  assert.equal(tags.length, 349714);
  assert.ok(tags.every((tag) => CATEGORIES[tag.category]));
  assert.equal(
    filterTags(tags, {
      query: "longhair",
      category: "0",
      theme: "appearance",
    })[0].name,
    "long_hair",
  );
});
