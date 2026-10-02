export const TAG_URL =
  "/tags/danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv";
export const CATEGORIES = {
  0: "Danbooru · Chung",
  1: "Danbooru · Họa sĩ",
  3: "Danbooru · Tác phẩm",
  4: "Danbooru · Nhân vật",
  5: "Danbooru · Metadata",
  7: "e621 · Chung",
  8: "e621 · Họa sĩ",
  9: "e621 · Người đóng góp",
  10: "e621 · Tác phẩm",
  11: "e621 · Nhân vật",
  12: "e621 · Loài",
  14: "e621 · Metadata",
  15: "e621 · Lore",
};
export const THEMES = {
  appearance: [
    "Ngoại hình",
    /(?:hair|eyes?|ears?|face|skin|fur|wings?|tail|horns?)/,
  ],
  clothing: [
    "Trang phục & phụ kiện",
    /(?:dress|shirt|skirt|uniform|clothes|clothing|hat|ribbon|armor|jewelry|boots?|gloves?)/,
  ],
  expression: [
    "Biểu cảm & tư thế",
    /(?:smile|blush|crying|laughing|looking|standing|sitting|lying|pose|running|wink)/,
  ],
  scenery: [
    "Bối cảnh & thiên nhiên",
    /(?:background|sky|cloud|forest|tree|flower|ocean|beach|city|street|indoors|outdoors|room|mountain|snow|rain)/,
  ],
  lighting: [
    "Ánh sáng & màu sắc",
    /(?:light|shadow|sunset|sunrise|night|glow|neon|monochrome|color|colour)/,
  ],
  composition: [
    "Bố cục & kỹ thuật",
    /(?:view|portrait|close.up|full.body|perspective|focus|sketch|painting|watercolor|resolution|highres|absurdres|digital)/,
  ],
};

export function parseTags(text) {
  const rows = [];
  let row = [],
    field = "",
    quoted = false;
  const finish = () => {
    row.push(field);
    field = "";
    const [name, category, count, aliases = ""] = row;
    if (name && /^\d+$/.test(category) && /^\d+$/.test(count)) {
      rows.push({
        name,
        category,
        count: Number(count),
        aliases,
        search: normalize(`${name},${aliases}`),
      });
    }
    row = [];
  };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (c === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((c === "\n" || c === "\r") && !quoted) {
      finish();
      if (c === "\r" && text[i + 1] === "\n") i++;
    } else field += c;
  }
  if (field || row.length) finish();
  return rows;
}
export function normalize(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/[_\s]+/g, " ");
}
export function filterTags(
  tags,
  { query = "", category = "", theme = "", sort = "popular" } = {},
) {
  const term = normalize(query);
  return tags
    .filter(
      (tag) =>
        (!category || tag.category === category) &&
        (!term || tag.search.includes(term)) &&
        (!theme ||
          (theme === "other"
            ? !Object.values(THEMES).some(([, re]) => re.test(tag.name))
            : THEMES[theme]?.[1].test(tag.name))),
    )
    .sort(
      sort === "name"
        ? (a, b) => a.name.localeCompare(b.name)
        : (a, b) => b.count - a.count || a.name.localeCompare(b.name),
    );
}
export function appendTag(prompt, name, limit) {
  if (prompt.split(",").some((value) => normalize(value) === normalize(name)))
    return { error: "Thẻ đã có trong prompt." };
  const text = prompt.trim().replace(/[,\s]+$/, "");
  const value = text ? `${text}, ${name}` : name;
  return value.length > limit
    ? { error: `Prompt vượt giới hạn ${limit} ký tự.` }
    : { value };
}
let cached;
export function loadTags() {
  if (!cached)
    cached = fetch(TAG_URL)
      .then(async (response) => {
        if (!response.ok) throw new Error("Không tải được kho thẻ.");
        const tags = parseTags(await response.text());
        if (!tags.length) throw new Error("Kho thẻ trống hoặc không hợp lệ.");
        return tags;
      })
      .catch((error) => {
        cached = undefined;
        throw error;
      });
  return cached;
}
