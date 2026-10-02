import React, { useDeferredValue, useMemo, useState } from "react";
import {
  appendTag,
  CATEGORIES,
  THEMES,
  filterTags,
  loadTags,
  normalize,
} from "../lib/tags";

export default function TagPicker({ settings, update }) {
  const [open, setOpen] = useState(false);
  const [tags, setTags] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [theme, setTheme] = useState("");
  const [sort, setSort] = useState("popular");
  const [target, setTarget] = useState("prompt");
  const [page, setPage] = useState(1);
  const deferredQuery = useDeferredValue(query);
  const results = useMemo(
    () =>
      filterTags(tags || [], { query: deferredQuery, category, theme, sort }),
    [tags, deferredQuery, category, theme, sort],
  );
  const selected = new Set(settings[target].split(",").map(normalize));
  async function load() {
    setLoading(true);
    setError("");
    try {
      setTags(await loadTags());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }
  function change(setter, value) {
    setter(value);
    setPage(1);
  }
  function add(name) {
    const result = appendTag(
      settings[target],
      name,
      target === "prompt" ? 2000 : 1500,
    );
    if (result.error) setMessage(result.error);
    else {
      update(target, result.value);
      setMessage(
        `Đã thêm ${name} vào ${target === "prompt" ? "prompt" : "negative prompt"}.`,
      );
    }
  }
  return (
    <div className="tag-picker">
      <button
        type="button"
        className="tag-toggle"
        aria-expanded={open}
        aria-controls="tag-browser"
        onClick={() => {
          setOpen(!open);
          if (!open && !tags && !loading) load();
        }}
      >
        ＋ Thêm thẻ từ kho Danbooru / e621 <span>{open ? "−" : "⌄"}</span>
      </button>
      {open && (
        <div id="tag-browser">
          <p className="tag-help">
            Nguồn CSV · 01/10/2026. Danh mục theo dữ liệu gốc; chủ đề được nhóm
            tự động theo tên thẻ, có thể chồng lặp. Đây là từ khóa gợi ý, không
            phải mô hình hay ảnh huấn luyện. Kho có thể chứa thẻ nhạy cảm.
          </p>
          <input
            aria-label="Tìm thẻ"
            placeholder="Tìm tên thẻ hoặc bí danh…"
            value={query}
            onChange={(e) => change(setQuery, e.target.value)}
          />
          <div className="tag-filters">
            <label>
              Danh mục
              <select
                value={category}
                onChange={(e) => change(setCategory, e.target.value)}
              >
                <option value="">Tất cả danh mục</option>
                {Object.entries(CATEGORIES).map(([id, label]) => (
                  <option key={id} value={id}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Chủ đề
              <select
                value={theme}
                onChange={(e) => change(setTheme, e.target.value)}
              >
                <option value="">Tất cả chủ đề</option>
                {Object.entries(THEMES).map(([id, [label]]) => (
                  <option key={id} value={id}>
                    {label}
                  </option>
                ))}
                <option value="other">Chưa phân nhóm</option>
              </select>
            </label>
            <label>
              Sắp xếp
              <select
                value={sort}
                onChange={(e) => change(setSort, e.target.value)}
              >
                <option value="popular">Phổ biến nhất</option>
                <option value="name">Tên A–Z</option>
              </select>
            </label>
            <label>
              Thêm vào
              <select
                value={target}
                onChange={(e) => {
                  setTarget(e.target.value);
                  setMessage("");
                }}
              >
                <option value="prompt">Prompt</option>
                <option value="negative_prompt">Negative prompt</option>
              </select>
            </label>
          </div>
          {loading && <p role="status">Đang tải kho thẻ…</p>}
          {error && (
            <div role="alert">
              {error}{" "}
              <button type="button" onClick={load}>
                Thử lại
              </button>
            </div>
          )}
          {tags && (
            <>
              <p className="tag-help">
                {results.length.toLocaleString("vi-VN")} thẻ phù hợp /{" "}
                {tags.length.toLocaleString("vi-VN")} thẻ
              </p>
              <div className="tag-results">
                {results.slice((page - 1) * 40, page * 40).map((tag) => (
                  <button
                    type="button"
                    key={`${tag.category}:${tag.name}`}
                    disabled={selected.has(normalize(tag.name))}
                    title={`${CATEGORIES[tag.category] || tag.category} · ${tag.count.toLocaleString("vi-VN")} lượt\nBí danh: ${tag.aliases || "—"}`}
                    onClick={() => add(tag.name)}
                  >
                    {selected.has(normalize(tag.name)) ? "✓ " : "+ "}
                    {tag.name}
                    <small>{tag.count.toLocaleString("vi-VN")}</small>
                  </button>
                ))}
              </div>
              {!results.length && (
                <p>Không tìm thấy thẻ. Thử từ khóa khác hoặc bỏ bộ lọc.</p>
              )}
              {results.length > 40 && (
                <div className="tag-pagination">
                  <button
                    type="button"
                    disabled={page === 1}
                    onClick={() => setPage(page - 1)}
                  >
                    Trước
                  </button>
                  <span>
                    {page} / {Math.ceil(results.length / 40)}
                  </span>
                  <button
                    type="button"
                    disabled={page * 40 >= results.length}
                    onClick={() => setPage(page + 1)}
                  >
                    Sau
                  </button>
                </div>
              )}
            </>
          )}
          <p role="status" className="tag-help">
            {message}
          </p>
        </div>
      )}
    </div>
  );
}
