// Nạp "thư viện prompt" từ file text: một file .txt/.md có dạng
//
//   === 50 PROMPTS – CHỦ ĐỀ ===
//   Tên tiếng Việt | Nội dung prompts tiếng Anh
//
//   PROMPT 01 - Tên tiếng Việt của prompt
//   Masterpiece, best quality, ultra-detailed anime style, ...
//   Negative: low quality, blurry
//
// Ngoài ra còn đọc được dạng bảng một dòng (Tên | Prompt), dạng JSON
// ([{ title, prompt }]) và file chỉ gồm các đoạn prompt cách nhau bởi dòng trống.

export const PROMPT_LIMIT = 2000;
export const NEGATIVE_LIMIT = 1500;

const LIBRARY_STORAGE_KEY = "mirai-prompt-library-v1";
const MAX_FILE_BYTES = 4_000_000;
const MAX_PERSIST_CHARS = 1_400_000;

const BANNER = /^\s*(?:={3,}|\*{3,})\s*(.+?)\s*(?:={3,}|\*{3,})\s*$/;
const MARKDOWN_TITLE = /^\s*#\s+(.+?)\s*$/;
// Bắt buộc phải có số thứ tự: "PROMPT 01 - Tên", "Prompt 12: Tên", "## prompt 3 – Tên".
// Dòng "Prompt: Masterpiece, ..." (không có số) được hiểu là nội dung prompt.
const PROMPT_HEADING =
  /^\s*(?:#{1,6}\s*)?\**\s*prompt\s*\.?\s*(\d{1,3})\s*\**\s*[-–—:.)\]]*\s*(.*?)\s*$/i;
const NUMBER_HEADING =
  /^\s*(?:#{1,6}\s*)?\[?(\d{1,3})\]?\s*[-–—.:/)]\s+(.+?)\s*$/;
const PIPE_ROW = /^\s*([^|]{1,90}?)\s*\|\s*([^|]{15,})\s*$/;
const TABLE_HEADER_ROW =
  /tên\s+tiếng\s+việt|nội\s+dung\s+prompt|prompt\s+tiếng\s+anh|vietnamese\s+(name|title)|prompt\s*\(\s*en\w*\s*\)|\btitle\b.*\bprompt\b/i;
const BULLET = /^[-*•·]\s+/;
const RULE_LINE = /^\s*(?:-{3,}|_{3,})\s*$/;

const DIRECTIVES = [
  {
    key: "negative",
    re: /^\s*(?:negative(?:\s*prompt)?|loại\s+trừ|prompt\s+trừ|trừ)\s*[:\-–—]\s*(.+)$/i,
  },
  {
    key: "steps",
    re: /^\s*(?:steps?|bước|số\s+bước)\s*[:\-–—]\s*(\d{1,3})\s*$/i,
  },
  {
    key: "cfg",
    re: /^\s*(?:cfg(?:\s*scale)?|guidance)\s*[:\-–—]\s*(\d{1,2}(?:[.,]\d)?)\s*$/i,
  },
  {
    key: "size",
    re: /^\s*(?:size|kích\s+thước|resolution|độ\s+phân\s+giải)\s*[:\-–—]\s*(\d{3,4})\s*[x×*]\s*(\d{3,4})\s*$/i,
  },
  { key: "seed", re: /^\s*seed\s*[:\-–—]\s*(-?\d{1,12})\s*$/i },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeText(value) {
  return String(value ?? "")
    .replace(/^\uFEFF/, "")
    .replace(/\r\n?/g, "\n")
    .replace(/\u00a0/g, " ");
}

function cleanTitle(value) {
  return String(value ?? "")
    .replace(BULLET, "")
    .replace(/^\s*prompt\s*[:\-–—]\s*/i, "")
    .replace(/\*{1,2}/g, "")
    .replace(/`/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 140);
}

function isTableHeaderRow(line) {
  return line.includes("|") && TABLE_HEADER_ROW.test(line);
}

function parseBody(lines) {
  const overrides = {};
  const parts = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let handled = false;
    for (const directive of DIRECTIVES) {
      const match = directive.re.exec(trimmed);
      if (!match) continue;
      handled = true;
      if (directive.key === "negative") {
        overrides.negative = match[1].trim().slice(0, NEGATIVE_LIMIT);
      } else if (directive.key === "steps") {
        overrides.steps = clamp(Number(match[1]), 1, 50);
      } else if (directive.key === "cfg") {
        overrides.cfg = clamp(Number(match[1].replace(",", ".")), 1, 20);
      } else if (directive.key === "size") {
        const width = Number(match[1]);
        const height = Number(match[2]);
        if (
          width >= 512 &&
          height >= 512 &&
          width <= 1344 &&
          height <= 1344 &&
          Math.max(width / height, height / width) <= 1.75
        ) {
          overrides.width = Math.round(width / 8) * 8;
          overrides.height = Math.round(height / 8) * 8;
        }
      } else if (directive.key === "seed") {
        const seed = Number(match[1]);
        if (seed === -1 || (seed >= 0 && seed <= 4_294_967_295))
          overrides.seed = Math.trunc(seed);
      }
      break;
    }
    if (handled) continue;
    parts.push(
      trimmed
        .replace(BULLET, "")
        .replace(/^\*{1,2}(.*)\*{1,2}$/, "$1")
        .replace(/`/g, ""),
    );
  }
  const prompt = parts
    .join(" ")
    .replace(/\s{2,}/g, " ")
    .replace(/\s+,/g, ",")
    .replace(/,\s*,+/g, ", ")
    .replace(/^\s*prompt\s*[:\-–—]\s*/i, "")
    .trim();
  return { prompt, overrides };
}

function buildItems(entries, name) {
  const items = [];
  for (const entry of entries) {
    const { prompt, overrides } = parseBody(entry.body);
    if (!prompt) continue;
    const truncated = prompt.length > PROMPT_LIMIT;
    const fallbackTitle = `${prompt.slice(0, 58)}${prompt.length > 58 ? "…" : ""}`;
    items.push({
      id: `${name}-${items.length + 1}`,
      index: items.length + 1,
      title: cleanTitle(entry.title) || fallbackTitle,
      prompt: truncated ? prompt.slice(0, PROMPT_LIMIT) : prompt,
      negative: overrides.negative || "",
      steps: overrides.steps,
      cfg: overrides.cfg,
      width: overrides.width,
      height: overrides.height,
      seed: overrides.seed,
      truncated,
    });
  }
  return items;
}

function fromJson(value, name) {
  const rows = Array.isArray(value) ? value : value?.items;
  if (!Array.isArray(rows)) return null;
  const entries = rows
    .filter((row) => row && typeof row === "object")
    .map((row) => ({
      title: row.title ?? row.name ?? row["tên tiếng việt"] ?? row.vi ?? "",
      body: [String(row.prompt ?? row.en ?? row.content ?? "")],
    }));
  if (!entries.length) return null;
  const items = buildItems(entries, name);
  return items.length
    ? {
        name,
        source: "json",
        description: "",
        items,
        truncated: items.filter((item) => item.truncated).length,
      }
    : null;
}

function parsePlainText(raw, name) {
  const lines = raw.split("\n");
  const tableMode =
    lines.filter(
      (line) => PIPE_ROW.test(line) && !isTableHeaderRow(line.trim()),
    ).length >= 3;

  let title = "";
  let description = "";
  const entries = [];
  let current = null;
  let expected = 1;
  const orphan = [];

  const flush = () => {
    if (current) {
      entries.push(current);
      current = null;
    }
  };
  const start = (heading) => {
    flush();
    current = { title: heading, body: [] };
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, "");
    const trimmed = line.trim();
    if (!trimmed) {
      if (current) current.body.push("");
      continue;
    }
    if (RULE_LINE.test(trimmed)) {
      if (current) current.body.push("");
      continue;
    }

    const banner = BANNER.exec(trimmed);
    if (banner && !entries.length && !current && !title) {
      title = cleanTitle(banner[1]);
      continue;
    }
    if (!title && !entries.length && !current) {
      const markdown = MARKDOWN_TITLE.exec(trimmed);
      if (markdown) {
        title = cleanTitle(markdown[1]);
        continue;
      }
    }
    if (isTableHeaderRow(trimmed)) continue;

    const promptHeading = PROMPT_HEADING.exec(trimmed);
    if (promptHeading) {
      const number = Number(promptHeading[1]);
      if (Number.isFinite(number)) expected = number + 1;
      start(promptHeading[2] || "");
      continue;
    }

    const numbered = NUMBER_HEADING.exec(trimmed);
    // Chỉ nhận "01 - Tên" khi số thứ tự khớp, để không nhầm với các dòng
    // prompt bắt đầu bằng con số (ví dụ "3.5 mm lens").
    if (numbered && Number(numbered[1]) === expected) {
      expected += 1;
      start(numbered[2]);
      continue;
    }

    if (tableMode && PIPE_ROW.test(trimmed)) {
      const [, left, right] = PIPE_ROW.exec(trimmed);
      flush();
      entries.push({ title: left, body: [right] });
      continue;
    }

    if (current) current.body.push(trimmed);
    else if (!description && trimmed.length > 12) orphan.push(trimmed);
  }
  flush();
  description = orphan.join(" ").slice(0, 300);

  if (!entries.length) {
    // Không có tiêu đề nào: coi mỗi đoạn văn (cách nhau bởi dòng trống) là một prompt.
    const paragraphs = raw.split(/\n\s*\n/).map((block) => block.trim());
    for (const block of paragraphs) {
      if (!block || BANNER.test(block) || isTableHeaderRow(block)) continue;
      const body = block.split("\n").map((part) => part.trim());
      entries.push({
        title: cleanTitle(body[0]).slice(0, 70),
        body,
      });
    }
  }

  const items = buildItems(entries, name);
  return {
    name: title || name,
    source: tableMode ? "table" : "text",
    description,
    items,
    truncated: items.filter((item) => item.truncated).length,
  };
}

export function parsePromptLibrary(text, { name = "Thư viện prompt" } = {}) {
  const raw = normalizeText(text);
  if (!raw.trim()) throw new Error("File trống, không có prompt nào.");
  const trimmed = raw.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const parsed = fromJson(JSON.parse(trimmed), name);
      if (parsed) return parsed;
    } catch {
      // Không phải JSON hợp lệ — thử đọc như text thường.
    }
  }
  const library = parsePlainText(raw, name);
  if (!library.items.length)
    throw new Error(
      "Không tìm thấy prompt nào. Mỗi prompt cần một dòng tiêu đề dạng “PROMPT 01 - Tên tiếng Việt” hoặc một dòng “Tên | Prompt”.",
    );
  return library;
}

export function filterLibraryItems(items, query) {
  const needle = stripDiacritics(
    String(query ?? "")
      .trim()
      .toLowerCase(),
  );
  if (!needle) return items;
  const words = needle.split(/\s+/).filter(Boolean);
  return items.filter((item) => {
    const haystack = stripDiacritics(
      `${item.index} ${item.title} ${item.prompt}`.toLowerCase(),
    );
    return words.every((word) => haystack.includes(word));
  });
}

export function stripDiacritics(value) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D");
}

export function libraryToText(library) {
  if (!library?.items?.length) return "";
  const rawName = String(library.name || "Thư viện prompt").trim();
  // Tên thư viện lấy từ file thường đã có dạng "50 PROMPTS – CHỦ ĐỀ".
  const name = /^\d+\s*prompt/i.test(rawName)
    ? rawName.replace(/^\d+\s*prompt\w*\s*[–—-]\s*/i, "")
    : rawName;
  const heading = `=== ${library.items.length} PROMPTS – ${name} ===`;
  const blocks = library.items.map((item, position) => {
    const number = String(position + 1).padStart(2, "0");
    const lines = [`PROMPT ${number} - ${item.title}`, item.prompt];
    if (item.negative) lines.push(`Negative: ${item.negative}`);
    if (item.steps) lines.push(`Steps: ${item.steps}`);
    if (item.cfg) lines.push(`CFG: ${item.cfg}`);
    if (item.width && item.height)
      lines.push(`Size: ${item.width} x ${item.height}`);
    if (item.seed !== undefined) lines.push(`Seed: ${item.seed}`);
    return lines.join("\n");
  });
  return `${heading}\n\nTên tiếng Việt | Nội dung prompts tiếng Anh\n\n${blocks.join("\n\n")}\n`;
}

export function readTextFile(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error("Chưa chọn file prompts."));
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      reject(new Error("File prompts quá lớn. Giới hạn 4 MB."));
      return;
    }
    if (typeof file.text === "function") {
      file
        .text()
        .then((text) => resolve(normalizeText(text)))
        .catch(() => reject(new Error("Không đọc được nội dung file.")));
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve(normalizeText(reader.result));
    reader.onerror = () => reject(new Error("Không đọc được nội dung file."));
    reader.readAsText(file, "utf-8");
  });
}

export function persistLibrary(library) {
  try {
    const payload = JSON.stringify(library);
    if (payload.length > MAX_PERSIST_CHARS) return false;
    localStorage.setItem(LIBRARY_STORAGE_KEY, payload);
    return true;
  } catch {
    return false;
  }
}

export function readPersistedLibrary() {
  try {
    const stored = localStorage.getItem(LIBRARY_STORAGE_KEY);
    if (!stored) return null;
    const parsed = JSON.parse(stored);
    if (
      !parsed ||
      !Array.isArray(parsed.items) ||
      !parsed.items.length ||
      !parsed.items.every((item) => typeof item?.prompt === "string")
    )
      return null;
    return {
      name: typeof parsed.name === "string" ? parsed.name : "Thư viện prompt",
      source: typeof parsed.source === "string" ? parsed.source : "text",
      description:
        typeof parsed.description === "string" ? parsed.description : "",
      items: parsed.items,
      truncated: Number(parsed.truncated) || 0,
    };
  } catch {
    return null;
  }
}

export function clearPersistedLibrary() {
  try {
    localStorage.removeItem(LIBRARY_STORAGE_KEY);
  } catch {
    /* chế độ ẩn danh */
  }
}
