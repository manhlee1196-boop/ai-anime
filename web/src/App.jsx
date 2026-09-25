import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  BadgeCheck,
  BookOpen,
  Brush,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cloud,
  Copy,
  Cpu,
  Dice5,
  Download,
  ExternalLink,
  Eye,
  Heart,
  Image as ImageIcon,
  ImagePlus,
  Info,
  KeyRound,
  Layers3,
  LockKeyhole,
  Maximize2,
  MoreHorizontal,
  PenLine,
  Plus,
  RefreshCcw,
  ScanFace,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import MaskEditor from "./components/MaskEditor";
import {
  deleteGeneration,
  listGenerations,
  saveGeneration,
  setFavorite,
} from "./lib/history";
import {
  downloadBlob,
  formatTime,
  getRandomSeed,
  prepareSource,
} from "./lib/images";
import {
  COLAB_URL,
  DEFAULTS,
  INSPIRATION,
  MODEL_INFO,
  SIZE_PRESETS,
  STYLES,
} from "./lib/presets";

const STORAGE_KEY = "mirai-settings-v1";
const MODES = [
  { id: "txt2img", label: "Văn bản → ảnh", icon: WandSparkles },
  { id: "img2img", label: "Ảnh → ảnh", icon: ImageIcon },
  { id: "inpaint", label: "Sửa vùng ảnh", icon: Brush },
];

function readSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (!saved || typeof saved !== "object" || Array.isArray(saved))
      return DEFAULTS;
    const validNumber = (key, min, max, integer = false) => {
      const value = saved[key];
      return typeof value === "number" &&
        Number.isFinite(value) &&
        value >= min &&
        value <= max &&
        (!integer || Number.isInteger(value))
        ? value
        : DEFAULTS[key];
    };
    const provider = saved.provider === "wai" ? "wai" : "cloudflare";
    const model =
      provider === "wai"
        ? "wai-v17"
        : ["sdxl-base", "sdxl-lightning"].includes(saved.model)
          ? saved.model
          : "sdxl-base";
    const width = validNumber("width", 512, 1344, true);
    const height = validNumber("height", 512, 1344, true);
    const dimensionsOk =
      width % 8 === 0 &&
      height % 8 === 0 &&
      width * height <= 1_500_000 &&
      Math.max(width / height, height / width) <= 1.75;
    const lora = (name) => ({
      enabled:
        typeof saved.loras?.[name]?.enabled === "boolean"
          ? saved.loras[name].enabled
          : DEFAULTS.loras[name].enabled,
      weight:
        typeof saved.loras?.[name]?.weight === "number" &&
        saved.loras[name].weight >= 0.1 &&
        saved.loras[name].weight <= 1
          ? saved.loras[name].weight
          : DEFAULTS.loras[name].weight,
    });
    return {
      ...DEFAULTS,
      provider,
      model,
      mode: MODES.some((mode) => mode.id === saved.mode)
        ? saved.mode
        : DEFAULTS.mode,
      prompt:
        typeof saved.prompt === "string"
          ? saved.prompt.slice(0, 2000)
          : DEFAULTS.prompt,
      negative_prompt:
        typeof saved.negative_prompt === "string"
          ? saved.negative_prompt.slice(0, 1500)
          : DEFAULTS.negative_prompt,
      width: dimensionsOk ? width : DEFAULTS.width,
      height: dimensionsOk ? height : DEFAULTS.height,
      steps: validNumber("steps", 1, provider === "wai" ? 45 : 20, true),
      cfg: validNumber("cfg", 1, 12),
      seed: validNumber("seed", -1, 4294967295, true),
      strength: validNumber("strength", 0.2, 0.95),
      count: validNumber("count", 1, 4, true),
      scheduler: ["euler_a", "dpmpp_2m"].includes(saved.scheduler)
        ? saved.scheduler
        : DEFAULTS.scheduler,
      clip_skip: [1, 2].includes(saved.clip_skip)
        ? saved.clip_skip
        : DEFAULTS.clip_skip,
      loras: { anatomy: lora("anatomy"), eyes: lora("eyes") },
    };
  } catch {
    return DEFAULTS;
  }
}

function IconLogo({ small = false }) {
  return (
    <span className={`brand-logo ${small ? "brand-logo-small" : ""}`}>
      <Sparkles size={small ? 19 : 22} strokeWidth={1.9} />
    </span>
  );
}

function FieldHead({ title, hint, right }) {
  return (
    <div className="field-head">
      <div>
        <span className="field-title">{title}</span>
        {hint && <span className="field-hint">{hint}</span>}
      </div>
      {right}
    </div>
  );
}

function RangeField({
  label,
  value,
  min,
  max,
  step,
  onChange,
  suffix = "",
  hint,
}) {
  return (
    <div className="range-field">
      <div className="range-header">
        <span>{label}</span>
        <span className="range-value">
          {value}
          {suffix}
        </span>
      </div>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {hint && <small>{hint}</small>}
    </div>
  );
}

function SectionLabel({ icon: Icon, title, number }) {
  return (
    <div className="section-label">
      <span className="section-icon">
        <Icon size={15} />
      </span>
      <span>{title}</span>
      {number && <span className="section-number">{number}</span>}
    </div>
  );
}

function saveFilename(record) {
  const extension = record.blob.type.includes("jpeg")
    ? "jpg"
    : record.blob.type.includes("webp")
      ? "webp"
      : "png";
  return `mirai_${record.model}_${record.seed}_${record.createdAt}.${extension}`;
}

export default function App() {
  const [settings, setSettings] = useState(readSettings);
  const [config, setConfig] = useState(null);
  const [accessCode, setAccessCode] = useState(() => {
    try {
      return sessionStorage.getItem("mirai-access") || "";
    } catch {
      return "";
    }
  });
  const [authOpen, setAuthOpen] = useState(false);
  const [authInput, setAuthInput] = useState("");
  const [source, setSource] = useState(null);
  const [mask, setMask] = useState(null);
  const [gallery, setGallery] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [toast, setToast] = useState(null);
  const [advanced, setAdvanced] = useState(true);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [lightbox, setLightbox] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [sourceBusy, setSourceBusy] = useState(false);
  const inputRef = useRef(null);
  const galleryUrls = useRef([]);
  const volatileRecords = useRef([]);
  const abortRef = useRef(null);
  const toastTimer = useRef(null);
  const active = gallery.find((item) => item.id === activeId) || null;
  const visibleGallery = favoritesOnly
    ? gallery.filter((item) => item.favorite)
    : gallery;
  const providerReady =
    settings.provider === "cloudflare" ? config?.cloudflare : config?.wai;
  const trulyReady = Boolean(
    providerReady && config?.configured && !config?.localPreview,
  );
  const editing = settings.mode !== "txt2img";
  const dimensions =
    editing && source
      ? { width: source.width, height: source.height }
      : { width: settings.width, height: settings.height };

  const notice = useCallback((message, type = "info") => {
    setToast({ message, type });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4800);
  }, []);

  const loadGallery = useCallback(async () => {
    try {
      const records = await listGenerations();
      const inMemory = volatileRecords.current;
      galleryUrls.current
        .filter((url) => !inMemory.some((item) => item.url === url))
        .forEach((url) => URL.revokeObjectURL(url));
      const urls = records.map((item) => URL.createObjectURL(item.blob));
      galleryUrls.current = [...urls, ...inMemory.map((item) => item.url)];
      setGallery(
        [
          ...records.map((item, index) => ({ ...item, url: urls[index] })),
          ...inMemory,
        ].sort((a, b) => b.createdAt - a.createdAt),
      );
    } catch {
      // Private browsing / denied IndexedDB: allow an in-memory session.
    }
  }, []);

  useEffect(() => {
    fetch("/api/config")
      .then((res) => res.json())
      .then(setConfig)
      .catch(() =>
        setConfig({
          cloudflare: false,
          wai: false,
          configured: false,
          localPreview: true,
        }),
      );
    loadGallery();
    return () => {
      galleryUrls.current.forEach((url) => URL.revokeObjectURL(url));
      clearTimeout(toastTimer.current);
    };
  }, [loadGallery]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* storage unavailable */
    }
  }, [settings]);

  const update = (key, value) =>
    setSettings((previous) => ({ ...previous, [key]: value }));
  const updateLora = (key, change) =>
    setSettings((previous) => ({
      ...previous,
      loras: {
        ...previous.loras,
        [key]: { ...previous.loras[key], ...change },
      },
    }));

  function saveAccessCode() {
    const value = authInput.trim();
    if (!value) return;
    setAccessCode(value);
    try {
      sessionStorage.setItem("mirai-access", value);
    } catch {
      /* private browsing */
    }
    setAuthOpen(false);
    notice("Đã lưu mã truy cập cho phiên này.", "success");
  }

  function selectProvider(provider) {
    setSettings((previous) => ({
      ...previous,
      provider,
      model: provider === "wai" ? "wai-v17" : "sdxl-base",
      steps: provider === "wai" ? 25 : 20,
    }));
  }

  function selectModel(model) {
    setSettings((previous) => ({
      ...previous,
      model,
      steps: MODEL_INFO[model].suggestedSteps,
    }));
  }

  function selectPreset(item) {
    setSettings((previous) => ({
      ...previous,
      width: item.width,
      height: item.height,
    }));
  }

  function addStyle(style) {
    if (settings.prompt.toLowerCase().includes(style.tag.toLowerCase())) {
      notice("Phong cách này đã có trong prompt.");
      return;
    }
    const combined =
      settings.prompt.trim().replace(/,?\s*$/, "") + ", " + style.tag;
    if (combined.length > 2000) {
      notice("Prompt đã đầy. Hãy rút ngắn trước khi thêm phong cách.", "error");
      return;
    }
    update("prompt", combined);
    notice(`Đã thêm phong cách ${style.name}.`, "success");
  }

  async function loadSource(input) {
    setSourceBusy(true);
    try {
      const prepared = await prepareSource(input);
      setSource(prepared);
      setMask(null);
      setSettings((previous) => ({
        ...previous,
        width: prepared.width,
        height: prepared.height,
      }));
      notice(
        `Đã nạp ${prepared.name} · ${prepared.width} × ${prepared.height}`,
        "success",
      );
      return true;
    } catch (error) {
      notice(error.message || "Không mở được ảnh.", "error");
      return false;
    } finally {
      setSourceBusy(false);
    }
  }

  async function useSampleAsSource(sample) {
    try {
      const response = await fetch(sample.image);
      if (!response.ok) throw new Error("Không mở được ảnh minh họa.");
      const blob = await response.blob();
      if (
        !(await loadSource(
          new File([blob], `${sample.title}.jpg`, {
            type: blob.type || "image/jpeg",
          }),
        ))
      )
        return;
      update("mode", "inpaint");
      document
        .getElementById("workspace")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      notice(error.message, "error");
    }
  }

  async function useRecordAsSource(record) {
    if (
      !(await loadSource(
        new File([record.blob], saveFilename(record), {
          type: record.blob.type || "image/png",
        }),
      ))
    )
      return;
    setSettings((previous) => ({
      ...previous,
      mode: "inpaint",
      prompt: record.prompt,
      negative_prompt: record.negative_prompt,
    }));
    setLightbox(false);
    document
      .getElementById("workspace")
      ?.scrollIntoView({ behavior: "smooth" });
  }

  async function generate() {
    if (busy) return;
    if (!settings.prompt.trim()) {
      notice("Hãy nhập ý tưởng vào ô Prompt trước.", "error");
      return;
    }
    if (!trulyReady) {
      notice(
        config?.localPreview
          ? "Đây là bản xem trước. Triển khai Worker để tạo ảnh thật."
          : "Nguồn tạo ảnh chưa sẵn sàng. Xem hướng dẫn cấu hình.",
        "error",
      );
      return;
    }
    if (!accessCode) {
      setAuthOpen(true);
      notice("Nhập mã truy cập bạn đã thiết lập cho ứng dụng.");
      return;
    }
    if (editing && !source) {
      notice("Tải một ảnh nguồn trước khi sửa hoặc biến đổi.", "error");
      return;
    }
    if (settings.mode === "inpaint" && !mask) {
      notice("Tô vùng cần sửa bằng cọ màu hồng hoặc tải mask PNG.", "error");
      return;
    }
    if (
      settings.provider === "wai" &&
      settings.loras.eyes.enabled &&
      !/\bperfect eyes\b/i.test(settings.prompt) &&
      settings.prompt.trim().length > 1986
    ) {
      notice(
        'Prompt WAI quá dài để thêm trigger "perfect eyes". Hãy rút ngắn prompt.',
        "error",
      );
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    let historyFailed = false;
    try {
      for (let i = 0; i < settings.count; i++) {
        const seed =
          settings.seed === -1 ? getRandomSeed() : (settings.seed + i) >>> 0;
        setProgress(`${i + 1}/${settings.count}`);
        const prompt =
          settings.provider === "wai" &&
          settings.loras.eyes.enabled &&
          !/\bperfect eyes\b/i.test(settings.prompt)
            ? `${settings.prompt.trim()}, perfect eyes`
            : settings.prompt.trim();
        const payload = {
          provider: settings.provider,
          model: settings.model,
          mode: settings.mode,
          prompt,
          negative_prompt: settings.negative_prompt.trim(),
          width: dimensions.width,
          height: dimensions.height,
          steps: settings.steps,
          cfg: settings.cfg,
          seed,
          strength: settings.strength,
          ...(settings.provider === "wai"
            ? {
                scheduler: settings.scheduler,
                clip_skip: settings.clip_skip,
                loras: settings.loras,
              }
            : {}),
          ...(editing ? { image_b64: source.dataURL } : {}),
          ...(settings.mode === "inpaint" ? { mask_b64: mask } : {}),
        };
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessCode}`,
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!response.ok) {
          const problem = await response.json().catch(() => ({}));
          if (response.status === 401) setAuthOpen(true);
          throw new Error(
            problem.error || `Yêu cầu thất bại (${response.status}).`,
          );
        }
        const imageBlob = await response.blob();
        if (!imageBlob.type.startsWith("image/") || !imageBlob.size)
          throw new Error("Máy chủ không trả về ảnh hợp lệ.");
        const record = {
          id: crypto.randomUUID(),
          blob: imageBlob,
          createdAt: Date.now(),
          favorite: false,
          prompt: payload.prompt,
          negative_prompt: payload.negative_prompt,
          provider: settings.provider,
          model: settings.model,
          mode: settings.mode,
          width: dimensions.width,
          height: dimensions.height,
          steps: settings.steps,
          cfg: settings.cfg,
          seed: Number(response.headers.get("X-Image-Seed")) || seed,
          strength: settings.strength,
          loras: settings.provider === "wai" ? settings.loras : null,
          duration: Number(response.headers.get("X-Generation-Ms") || 0),
        };
        try {
          await saveGeneration(record);
          await loadGallery();
        } catch {
          historyFailed = true;
          const url = URL.createObjectURL(imageBlob);
          const ephemeral = { ...record, url, volatile: true };
          galleryUrls.current.push(url);
          volatileRecords.current.unshift(ephemeral);
          setGallery((previous) => [ephemeral, ...previous]);
        }
        setActiveId(record.id);
      }
      notice(
        historyFailed
          ? "Ảnh đã tạo nhưng không lưu được lâu dài. Hãy tải ảnh về trước khi đóng trang."
          : `Đã tạo ${settings.count} ảnh. Ảnh nằm trong bộ sưu tập trên trình duyệt này.`,
        historyFailed ? "info" : "success",
      );
    } catch (error) {
      if (error.name !== "AbortError")
        notice(error.message || "Có lỗi khi tạo ảnh.", "error");
      else
        notice(
          "Đã dừng chờ ảnh. Yêu cầu đang xử lý trên máy chủ có thể vẫn tiêu tốn lượt tạo.",
        );
    } finally {
      abortRef.current = null;
      setBusy(false);
      setProgress("");
    }
  }

  async function toggleFavorite(record) {
    if (record.volatile) {
      volatileRecords.current = volatileRecords.current.map((item) =>
        item.id === record.id ? { ...item, favorite: !item.favorite } : item,
      );
      setGallery((previous) =>
        previous.map((item) =>
          item.id === record.id ? { ...item, favorite: !item.favorite } : item,
        ),
      );
      return;
    }
    try {
      await setFavorite(record, !record.favorite);
      await loadGallery();
    } catch {
      notice("Không cập nhật được ảnh yêu thích.", "error");
    }
  }

  async function removeRecord(record) {
    if (!window.confirm("Xóa ảnh này khỏi lịch sử trên trình duyệt?")) return;
    try {
      if (record.volatile) {
        volatileRecords.current = volatileRecords.current.filter(
          (item) => item.id !== record.id,
        );
        galleryUrls.current = galleryUrls.current.filter(
          (url) => url !== record.url,
        );
        URL.revokeObjectURL(record.url);
        setGallery((previous) =>
          previous.filter((item) => item.id !== record.id),
        );
      } else {
        await deleteGeneration(record.id);
        await loadGallery();
      }
      if (activeId === record.id) setActiveId(null);
      setLightbox(false);
      notice("Đã xóa ảnh khỏi lịch sử.");
    } catch {
      notice(
        "Không xóa được ảnh. Hãy kiểm tra quyền lưu trữ của trình duyệt.",
        "error",
      );
    }
  }

  async function copyPrompt(text) {
    try {
      await navigator.clipboard.writeText(text);
      notice("Đã sao chép prompt.", "success");
    } catch {
      notice("Không sao chép được. Hãy sao chép thủ công.", "error");
    }
  }

  useEffect(() => {
    const handler = (event) => {
      if (event.key === "Escape") {
        setLightbox(false);
        setAuthOpen(false);
        setHelpOpen(false);
      }
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        generate();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const modelLabel = MODEL_INFO[settings.model]?.name || settings.model;
  const statusLabel = config?.localPreview
    ? "Bản xem trước"
    : trulyReady
      ? "Đã cấu hình"
      : settings.provider === "wai" && config?.configured
        ? "Cần máy GPU riêng"
        : "Cần cấu hình";
  const statusClass = config?.localPreview
    ? "status-preview"
    : trulyReady
      ? "status-ready"
      : "status-offline";

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-left">
          <a
            className="brand"
            href="#workspace"
            aria-label="Mirai Studio, về trang chính"
          >
            <IconLogo />
            <span className="brand-name">
              mirai<span className="brand-dot">.</span>
            </span>
            <span className="brand-studio">STUDIO</span>
          </a>
          <span className="header-separator" />
          <span className="header-product">
            Anime image workspace <span>✦</span>
          </span>
        </div>
        <nav className="top-nav" aria-label="Điều hướng">
          <a className="nav-active" href="#workspace">
            Studio
          </a>
          <a href="#inspiration">Cảm hứng</a>
          <button type="button" onClick={() => setHelpOpen(true)}>
            Hướng dẫn
          </button>
        </nav>
        <div className="topbar-right">
          <span className={`connection-status ${statusClass}`}>
            <span className="status-dot" />
            {statusLabel}
          </span>
          <button
            className="access-button"
            onClick={() => {
              setAuthInput(accessCode);
              setAuthOpen(true);
            }}
            type="button"
          >
            <KeyRound size={16} />
            <span>Mã truy cập</span>
          </button>
          <span className="user-avatar">M</span>
        </div>
      </header>

      <div className="app-layout" id="workspace">
        <aside className="control-panel">
          <div className="controls-heading">
            <div>
              <span className="eyebrow">
                CREATIVE CONTROLS <span className="eyebrow-star">✦</span>
              </span>
              <h2>Tạo một điều kỳ diệu.</h2>
              <p>Mọi ý tưởng đều bắt đầu từ đây.</p>
            </div>
            <button
              type="button"
              className="reset-icon"
              title="Đặt lại thông số"
              onClick={() => {
                setSettings(DEFAULTS);
                setSource(null);
                setMask(null);
                notice("Đã đặt lại thông số mặc định.");
              }}
            >
              <RefreshCcw size={17} />
            </button>
          </div>
          <div className="control-scroll">
            <section className="control-section provider-section">
              <SectionLabel
                icon={Layers3}
                title="Nền tảng tạo ảnh"
                number="01"
              />
              <div className="provider-switch">
                <button
                  type="button"
                  className={
                    settings.provider === "cloudflare" ? "is-active" : ""
                  }
                  onClick={() => selectProvider("cloudflare")}
                >
                  <Cloud size={17} /> Cloudflare AI
                </button>
                <button
                  type="button"
                  className={settings.provider === "wai" ? "is-active" : ""}
                  onClick={() => selectProvider("wai")}
                >
                  <Cpu size={17} /> WAI · GPU
                </button>
              </div>
              <p className="tiny-help">
                {settings.provider === "cloudflare"
                  ? "Chạy trên Workers AI. Đây là SDXL của Cloudflare, không phải checkpoint WAI."
                  : "Giữ đúng checkpoint WAI v17 + LoRA đã xác thực. Cần kết nối máy GPU bên ngoài."}
              </p>
            </section>

            <section className="control-section">
              <SectionLabel
                icon={Sparkles}
                title="Chế độ sáng tạo"
                number="02"
              />
              <div className="mode-select">
                {MODES.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    className={settings.mode === item.id ? "is-active" : ""}
                    onClick={() => update("mode", item.id)}
                  >
                    <item.icon size={16} />
                    <span>{item.label}</span>
                  </button>
                ))}
              </div>
            </section>

            <section className="control-section prompt-section">
              <SectionLabel
                icon={PenLine}
                title="Ý tưởng của bạn"
                number="03"
              />
              <FieldHead
                title="Prompt"
                right={
                  <button
                    type="button"
                    className="subtle-action"
                    onClick={() => {
                      const sample =
                        INSPIRATION[
                          Math.floor(Math.random() * INSPIRATION.length)
                        ];
                      update("prompt", sample.prompt);
                      notice("Đã chọn một ý tưởng ngẫu nhiên.");
                    }}
                  >
                    <Dice5 size={14} /> Gợi ý
                  </button>
                }
              />
              <textarea
                aria-label="Prompt tạo ảnh"
                spellCheck="false"
                rows="5"
                value={settings.prompt}
                maxLength={2000}
                onChange={(event) => update("prompt", event.target.value)}
                placeholder="Mô tả nhân vật, khung cảnh, ánh sáng, phong cách…"
              />
              <div className="input-footer">
                <span>Hãy miêu tả điều bạn muốn nhìn thấy</span>
                <span>{settings.prompt.length}/2000</span>
              </div>
              <div className="styles-label">
                THỬ PHONG CÁCH <Sparkles size={12} />
              </div>
              <div className="style-chips">
                {STYLES.map((style) => (
                  <button
                    type="button"
                    key={style.name}
                    onClick={() => addStyle(style)}
                  >
                    {style.name} <Plus size={12} />
                  </button>
                ))}
              </div>
            </section>

            <section className="control-section model-section">
              <SectionLabel
                icon={Settings2}
                title="Model & thông số"
                number="04"
              />
              <FieldHead
                title="Mô hình"
                right={
                  <span className="model-caption">
                    {settings.provider === "wai" ? "GPU RIÊNG" : "WORKERS AI"}
                  </span>
                }
              />
              <div className="select-wrap">
                <select
                  aria-label="Chọn mô hình"
                  value={settings.model}
                  onChange={(event) => selectModel(event.target.value)}
                >
                  {settings.provider === "cloudflare" ? (
                    <>
                      <option value="sdxl-base">
                        SDXL Base 1.0 — chi tiết
                      </option>
                      <option value="sdxl-lightning">
                        SDXL Lightning — tốc độ
                      </option>
                    </>
                  ) : (
                    <option value="wai-v17">WAI-illustrious v17 — anime</option>
                  )}
                </select>
                <ChevronDown size={16} />
              </div>
              <p className="tiny-help model-description">
                {MODEL_INFO[settings.model]?.description}
              </p>
              <FieldHead
                title="Tỷ lệ khung hình"
                right={
                  <span className="dimension-pill">
                    {dimensions.width} × {dimensions.height}
                  </span>
                }
              />
              <div className="size-grid">
                {SIZE_PRESETS.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    disabled={editing && !!source}
                    className={
                      settings.width === item.width &&
                      settings.height === item.height
                        ? "selected"
                        : ""
                    }
                    onClick={() => selectPreset(item)}
                  >
                    <span className={`ratio-icon ratio-${item.icon}`} />
                    <b>{item.label}</b>
                    <small>{item.ratio}</small>
                  </button>
                ))}
              </div>
              {editing && source && (
                <p className="tiny-help source-dimension-note">
                  Kích thước đang theo ảnh nguồn đã tải lên.
                </p>
              )}
              <details className="custom-size">
                <summary>
                  Tuỳ chỉnh kích thước <ChevronDown size={14} />
                </summary>
                <div>
                  <label>
                    Rộng{" "}
                    <input
                      aria-label="Chiều rộng tùy chỉnh"
                      type="number"
                      min="512"
                      max="1344"
                      step="8"
                      disabled={editing && !!source}
                      value={settings.width}
                      onChange={(event) =>
                        update("width", Number(event.target.value))
                      }
                    />
                  </label>
                  <span>×</span>
                  <label>
                    Cao{" "}
                    <input
                      aria-label="Chiều cao tùy chỉnh"
                      type="number"
                      min="512"
                      max="1344"
                      step="8"
                      disabled={editing && !!source}
                      value={settings.height}
                      onChange={(event) =>
                        update("height", Number(event.target.value))
                      }
                    />
                  </label>
                </div>
                <small>Tối đa 1,5 MP; hai cạnh chia hết cho 8.</small>
              </details>
              <div className="range-stack">
                <RangeField
                  label="Số bước (steps)"
                  value={settings.steps}
                  min={1}
                  max={settings.provider === "wai" ? 45 : 20}
                  step={1}
                  onChange={(value) => update("steps", value)}
                  hint={
                    settings.model === "sdxl-lightning"
                      ? "Lightning thường đủ đẹp ở 4–8 bước."
                      : undefined
                  }
                />
                <RangeField
                  label="CFG / Guidance"
                  value={settings.cfg}
                  min={1}
                  max={12}
                  step={0.5}
                  onChange={(value) => update("cfg", value)}
                />
              </div>
            </section>

            {editing && (
              <section className="control-section upload-section">
                <SectionLabel
                  icon={ImagePlus}
                  title={
                    settings.mode === "inpaint"
                      ? "Ảnh cần sửa"
                      : "Ảnh tham chiếu"
                  }
                  number="05"
                />
                <button
                  type="button"
                  className={`upload-area ${source ? "has-source" : ""}`}
                  onClick={() => inputRef.current?.click()}
                  disabled={sourceBusy}
                >
                  <span className="upload-icon">
                    {source ? <CheckCircle2 size={22} /> : <Upload size={22} />}
                  </span>
                  <strong>{source ? source.name : "Chọn ảnh của bạn"}</strong>
                  <span>
                    {source
                      ? `${source.width} × ${source.height} · Nhấn để thay ảnh`
                      : "PNG, JPG, WebP · tối đa 12 MB"}
                  </span>
                </button>
                <input
                  ref={inputRef}
                  hidden
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (file) loadSource(file);
                  }}
                />
                {source && (
                  <button
                    className="remove-source"
                    onClick={() => {
                      setSource(null);
                      setMask(null);
                    }}
                    type="button"
                  >
                    <X size={13} /> Xóa ảnh nguồn
                  </button>
                )}
                <RangeField
                  label="Denoise strength"
                  value={settings.strength}
                  min={0.2}
                  max={0.95}
                  step={0.05}
                  onChange={(value) => update("strength", value)}
                  hint="Thấp giữ ảnh gốc; cao thay đổi mạnh hơn."
                />
              </section>
            )}

            {settings.provider === "wai" && (
              <section className="control-section lora-section">
                <SectionLabel
                  icon={ScanFace}
                  title="LoRA cải thiện chi tiết"
                  number="✦"
                />
                <p className="tiny-help">
                  Chỉ áp dụng trên máy GPU đã nạp đúng WAI và file LoRA được
                  kiểm tra SHA-256. Worker không tự tải hay kiểm tra trọng số.
                </p>
                <div className="lora-card">
                  <div className="lora-card-top">
                    <span className="lora-icon hand-icon">
                      <Brush size={16} />
                    </span>
                    <div>
                      <b>
                        Anatomy Helper <span>v1</span>
                      </b>
                      <small>Tay · chân · tư thế</small>
                    </div>
                    <label className="switch" aria-label="Bật LoRA anatomy">
                      <input
                        type="checkbox"
                        checked={settings.loras.anatomy.enabled}
                        onChange={(event) =>
                          updateLora("anatomy", {
                            enabled: event.target.checked,
                          })
                        }
                      />
                      <span />
                    </label>
                  </div>
                  {settings.loras.anatomy.enabled && (
                    <RangeField
                      label="Cường độ"
                      value={settings.loras.anatomy.weight}
                      min={0.1}
                      max={1}
                      step={0.05}
                      onChange={(weight) => updateLora("anatomy", { weight })}
                    />
                  )}
                </div>
                <div className="lora-card">
                  <div className="lora-card-top">
                    <span className="lora-icon eye-icon">
                      <Eye size={17} />
                    </span>
                    <div>
                      <b>
                        Perfect Eyes <span>v1</span>
                      </b>
                      <small>Chi tiết đôi mắt</small>
                    </div>
                    <label className="switch" aria-label="Bật LoRA mắt">
                      <input
                        type="checkbox"
                        checked={settings.loras.eyes.enabled}
                        onChange={(event) =>
                          updateLora("eyes", { enabled: event.target.checked })
                        }
                      />
                      <span />
                    </label>
                  </div>
                  {settings.loras.eyes.enabled && (
                    <RangeField
                      label="Cường độ"
                      value={settings.loras.eyes.weight}
                      min={0.1}
                      max={1}
                      step={0.05}
                      onChange={(weight) => updateLora("eyes", { weight })}
                    />
                  )}
                </div>
              </section>
            )}

            <section className="control-section advanced-section">
              <button
                type="button"
                className="advanced-toggle"
                onClick={() => setAdvanced(!advanced)}
              >
                <SectionLabel
                  icon={SlidersHorizontal}
                  title="Tuỳ chọn nâng cao"
                  number={advanced ? "−" : "+"}
                />
              </button>
              {advanced && (
                <div className="advanced-content">
                  <FieldHead
                    title="Negative prompt"
                    hint="Những điều bạn không muốn xuất hiện"
                  />
                  <textarea
                    aria-label="Negative prompt"
                    rows="3"
                    value={settings.negative_prompt}
                    maxLength={1500}
                    onChange={(event) =>
                      update("negative_prompt", event.target.value)
                    }
                  />
                  {settings.provider === "wai" && (
                    <>
                      <FieldHead title="Sampler" />
                      <div className="select-wrap">
                        <select
                          aria-label="Sampler"
                          value={settings.scheduler}
                          onChange={(event) =>
                            update("scheduler", event.target.value)
                          }
                        >
                          <option value="euler_a">Euler a</option>
                          <option value="dpmpp_2m">DPM++ 2M</option>
                        </select>
                        <ChevronDown size={15} />
                      </div>
                      <FieldHead title="CLIP skip" />
                      <div className="select-wrap">
                        <select
                          aria-label="CLIP skip"
                          value={settings.clip_skip}
                          onChange={(event) =>
                            update("clip_skip", Number(event.target.value))
                          }
                        >
                          <option value={1}>1</option>
                          <option value={2}>2</option>
                        </select>
                        <ChevronDown size={15} />
                      </div>
                    </>
                  )}
                  <FieldHead
                    title="Seed"
                    right={
                      <button
                        type="button"
                        className="subtle-action"
                        onClick={() => update("seed", -1)}
                      >
                        <Dice5 size={14} /> Ngẫu nhiên
                      </button>
                    }
                  />
                  <input
                    aria-label="Seed"
                    type="number"
                    min="-1"
                    max="4294967295"
                    value={settings.seed}
                    onChange={(event) =>
                      update("seed", Number(event.target.value))
                    }
                  />
                  <p className="tiny-help">
                    -1 = ngẫu nhiên. Seed cố định giúp lặp lại điều kiện tạo
                    ảnh.
                  </p>
                  <FieldHead
                    title="Số ảnh"
                    hint="Tạo lần lượt, mỗi ảnh có seed riêng"
                  />
                  <div className="count-select">
                    {[1, 2, 3, 4].map((number) => (
                      <button
                        type="button"
                        key={number}
                        className={settings.count === number ? "selected" : ""}
                        onClick={() => update("count", number)}
                      >
                        {number}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </div>
          <div className="create-footer">
            <div className="create-estimate">
              <span>
                <Sparkles size={13} /> {settings.count} ảnh · {dimensions.width}{" "}
                × {dimensions.height}
              </span>
              <span>{settings.steps} bước</span>
            </div>
            <button
              className="generate-button"
              type="button"
              onClick={generate}
              disabled={busy}
            >
              <Sparkles size={19} />
              {busy
                ? `Đang tạo ${progress}...`
                : trulyReady
                  ? "Tạo ảnh ngay"
                  : "Kết nối để tạo ảnh"}
              <ArrowRight size={19} />
            </button>
            {busy ? (
              <button
                type="button"
                className="cancel-action"
                onClick={() => abortRef.current?.abort()}
              >
                Dừng chờ tạo ảnh
              </button>
            ) : (
              <p>Ctrl / ⌘ + Enter để tạo nhanh</p>
            )}
          </div>
        </aside>

        <main className="workspace-main">
          <div className="page-intro">
            <div>
              <div className="page-breadcrumb">
                KHÔNG GIAN SÁNG TẠO <ChevronRight size={12} />{" "}
                <span>STUDIO</span>
              </div>
              <h1>
                Để trí tưởng tượng
                <br />
                <em>lên tiếng.</em>
              </h1>
              <p>Một thế giới anime mới, bắt đầu từ ý tưởng của bạn.</p>
            </div>
            <div className="page-intro-action">
              <span className="live-spark">✳</span>
              <div>
                <strong>Không giới hạn cảm hứng</strong>
                <small>Tạo · Biến đổi · Chỉnh sửa</small>
              </div>
            </div>
          </div>

          {config?.localPreview && (
            <div className="preview-banner">
              <Info size={17} />
              <span>
                Đây là <strong>bản xem trước giao diện</strong>. Ảnh bên dưới là
                minh họa; để tạo ảnh thật, triển khai Worker trên tài khoản
                Cloudflare và đặt mã truy cập.
              </span>
              <button type="button" onClick={() => setHelpOpen(true)}>
                Cách triển khai <ArrowUpRight size={13} />
              </button>
            </div>
          )}
          {config && !config.localPreview && !config.configured && (
            <div className="preview-banner">
              <LockKeyhole size={17} />
              <span>
                Worker chưa có <strong>APP_ACCESS_TOKEN</strong>. Tạo secret để
                bảo vệ chi phí tạo ảnh.
              </span>
              <button type="button" onClick={() => setHelpOpen(true)}>
                Xem hướng dẫn <ArrowUpRight size={13} />
              </button>
            </div>
          )}

          {settings.mode === "inpaint" && source ? (
            <div className="editor-stage">
              <div className="stage-topline">
                <div>
                  <span className="eyebrow">INPAINT · SỬA VÙNG ẢNH</span>
                  <h3>Điều chỉnh từng chi tiết</h3>
                  <p>
                    Tô vùng tay, chân hoặc mắt cần sửa; mask hướng dẫn model tập
                    trung vào vùng đó.
                  </p>
                </div>
                <span className="stage-icon">
                  <Brush size={23} />
                </span>
              </div>
              <MaskEditor
                key={source.dataURL.slice(0, 64) + source.name}
                source={source}
                onChange={setMask}
                onError={(message) => notice(message, "error")}
              />
            </div>
          ) : settings.mode === "img2img" && source ? (
            <div className="source-stage">
              <img src={source.dataURL} alt="Ảnh tham chiếu đã chọn" />
              <div className="source-stage-copy">
                <span className="eyebrow">IMAGE TO IMAGE</span>
                <h3>Biến đổi theo cách của bạn</h3>
                <p>
                  Ảnh nguồn đã sẵn sàng. Chỉnh prompt và strength, rồi nhấn tạo
                  ảnh.
                </p>
                <span>
                  {source.width} × {source.height} px
                </span>
              </div>
            </div>
          ) : active ? (
            <div className="result-stage">
              <div className="result-stage-bar">
                <div>
                  <span className="result-live-dot" /> KẾT QUẢ MỚI NHẤT{" "}
                  <span className="result-separator">/</span>{" "}
                  {MODEL_INFO[active.model]?.name || active.model}
                </div>
                <span>SEED {active.seed}</span>
              </div>
              <div className="result-image-area">
                <img
                  src={active.url}
                  alt={`Ảnh anime đã tạo: ${active.prompt.slice(0, 90)}`}
                />
                <button
                  type="button"
                  className="stage-expand"
                  title="Xem toàn màn hình"
                  onClick={() => setLightbox(true)}
                >
                  <Maximize2 size={19} />
                </button>
              </div>
              <div className="result-actions">
                <span>
                  <CheckCircle2 size={16} />{" "}
                  {active.volatile
                    ? "Chưa lưu lâu dài — hãy tải ảnh về"
                    : "Đã lưu trong trình duyệt"}
                </span>
                <div>
                  <button
                    type="button"
                    onClick={() => toggleFavorite(active)}
                    title="Yêu thích"
                  >
                    <Heart
                      size={17}
                      fill={active.favorite ? "currentColor" : "none"}
                    />
                  </button>
                  <button
                    type="button"
                    title="Sửa ảnh này"
                    onClick={() => useRecordAsSource(active)}
                  >
                    <Brush size={17} />
                  </button>
                  <button
                    type="button"
                    title="Tải ảnh về"
                    onClick={() =>
                      downloadBlob(active.blob, saveFilename(active))
                    }
                  >
                    <Download size={17} />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="hero-stage">
              <div className="hero-copy">
                <span className="hero-kicker">
                  <span>✦</span> CREATE SOMETHING BEAUTIFUL
                </span>
                <h2>
                  Mỗi bức ảnh
                  <br />
                  là một <em>thế giới.</em>
                </h2>
                <p>
                  Biến câu chuyện trong đầu bạn thành một khung hình thật đẹp.
                  Bắt đầu với prompt ở bên trái.
                </p>
                <button
                  type="button"
                  onClick={() =>
                    document
                      .querySelector('[aria-label="Prompt tạo ảnh"]')
                      ?.focus()
                  }
                >
                  Bắt đầu sáng tạo <ArrowUpRight size={17} />
                </button>
                <div className="hero-bottom">
                  <span>01 / 03</span>
                  <div className="hero-progress">
                    <span />
                  </div>
                  <span>ILLUSTRATION SERIES</span>
                </div>
              </div>
              <div className="hero-art">
                <img
                  src="/samples/sakura-night.jpg"
                  alt="Tranh anime cô gái dưới hoa anh đào, ảnh minh họa giao diện"
                />
                <span className="hero-image-note">
                  ẢNH MINH HỌA · KHÔNG TẠO TRONG ỨNG DỤNG
                </span>
              </div>
              <div className="hero-orbit hero-orbit-one" />
              <div className="hero-orbit hero-orbit-two" />
            </div>
          )}

          <div className="workspace-stats">
            <div>
              <span className="stat-bubble purple">
                <WandSparkles size={18} />
              </span>
              <span>
                <b>Tạo từ ý tưởng</b>
                <small>Text → Image</small>
              </span>
            </div>
            <div>
              <span className="stat-bubble pink">
                <ImageIcon size={18} />
              </span>
              <span>
                <b>Biến đổi ảnh</b>
                <small>Image → Image</small>
              </span>
            </div>
            <div>
              <span className="stat-bubble blue">
                <Brush size={18} />
              </span>
              <span>
                <b>Sửa đúng vùng</b>
                <small>Inpainting mask</small>
              </span>
            </div>
          </div>

          <section className="inspiration-section" id="inspiration">
            <div className="content-heading">
              <div>
                <span className="eyebrow">GÓC CẢM HỨNG</span>
                <h2>
                  Ý tưởng chờ bạn khám phá <span>✦</span>
                </h2>
                <p>
                  Chọn một khung cảnh để bắt đầu — đây là ảnh minh họa, không
                  phải ảnh vừa tạo.
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  update(
                    "prompt",
                    INSPIRATION[Math.floor(Math.random() * INSPIRATION.length)]
                      .prompt,
                  );
                  notice("Đã chọn ý tưởng ngẫu nhiên.", "success");
                }}
              >
                Đổi gợi ý <RefreshCcw size={15} />
              </button>
            </div>
            <div className="inspiration-grid">
              {INSPIRATION.map((item, index) => (
                <article className="inspiration-card" key={item.title}>
                  <img
                    src={item.image}
                    loading="lazy"
                    alt={`Ảnh minh họa ${item.title}`}
                  />
                  <div className="inspiration-overlay">
                    <span>0{index + 1} / SAMPLE</span>
                    <h3>{item.title}</h3>
                    <p>{item.subtitle}</p>
                    <div className="sample-actions">
                      <button
                        type="button"
                        onClick={() => {
                          setSettings((previous) => ({
                            ...previous,
                            prompt: item.prompt,
                            width: index === 1 ? 1216 : 832,
                            height: index === 1 ? 832 : 1216,
                          }));
                          notice(
                            "Đã lấy prompt mẫu. Chỉnh lại trước khi tạo ảnh.",
                            "success",
                          );
                        }}
                      >
                        Dùng prompt <ArrowUpRight size={14} />
                      </button>
                      <button
                        type="button"
                        title="Dùng ảnh này để thử tô mask"
                        onClick={() => useSampleAsSource(item)}
                      >
                        <Brush size={14} />
                      </button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="gallery-section" id="gallery">
            <div className="content-heading">
              <div>
                <span className="eyebrow">BỘ SƯU TẬP CỦA BẠN</span>
                <h2>
                  Những khoảnh khắc đã tạo <span>✦</span>
                </h2>
                <p>
                  Ảnh được lưu cục bộ trong trình duyệt này, không tự đăng công
                  khai.
                </p>
              </div>
              <div className="gallery-filter">
                <button
                  type="button"
                  className={!favoritesOnly ? "selected" : ""}
                  onClick={() => setFavoritesOnly(false)}
                >
                  Tất cả <span>{gallery.length}</span>
                </button>
                <button
                  type="button"
                  className={favoritesOnly ? "selected" : ""}
                  onClick={() => setFavoritesOnly(true)}
                >
                  <Heart size={14} /> Yêu thích
                </button>
              </div>
            </div>
            {visibleGallery.length ? (
              <div className="gallery-grid">
                {visibleGallery.map((record) => (
                  <button
                    className={`gallery-card ${record.id === activeId ? "selected" : ""}`}
                    type="button"
                    key={record.id}
                    onClick={() => {
                      setActiveId(record.id);
                      setSettings((previous) => ({
                        ...previous,
                        mode: "txt2img",
                      }));
                      document
                        .getElementById("workspace")
                        ?.scrollIntoView({ behavior: "smooth" });
                    }}
                  >
                    <img
                      src={record.url}
                      loading="lazy"
                      alt={`Ảnh đã tạo với seed ${record.seed}`}
                    />
                    <span className="gallery-label">
                      {MODEL_INFO[record.model]?.name || record.model}
                      <ArrowUpRight size={14} />
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty-gallery">
                <span>
                  <ImageIcon size={24} />
                </span>
                <strong>
                  {favoritesOnly
                    ? "Chưa có ảnh yêu thích"
                    : "Bộ sưu tập đang chờ bạn"}
                </strong>
                <p>
                  {favoritesOnly
                    ? "Nhấn biểu tượng trái tim trên một ảnh để thêm vào đây."
                    : "Ảnh bạn tạo sẽ xuất hiện ở đây. Hãy bắt đầu với một ý tưởng!"}
                </p>
              </div>
            )}
          </section>
          <footer className="footer">
            <div className="brand footer-brand">
              <IconLogo small />
              <span className="brand-name">
                mirai<span className="brand-dot">.</span>
              </span>
              <span className="brand-studio">STUDIO</span>
            </div>
            <span>Made for the worlds you imagine.</span>
            <a href={COLAB_URL} target="_blank" rel="noopener noreferrer">
              Notebook WAI <ExternalLink size={13} />
            </a>
          </footer>
        </main>

        <aside className="inspector-panel">
          <div className="inspector-header">
            <span className="eyebrow">STUDIO OVERVIEW</span>
            <MoreHorizontal size={20} />
          </div>
          <div className="inspector-block current-model">
            <div className="inspector-block-title">
              Phiên làm việc <span className="pulse-dot" />
            </div>
            <span className="workspace-avatar">
              <Sparkles size={23} />
            </span>
            <h3>Anime Studio</h3>
            <p>Không gian sáng tạo của riêng bạn.</p>
            <div className="session-details">
              <span>
                Nguồn tạo ảnh{" "}
                <b>
                  {settings.provider === "wai" ? "GPU riêng" : "Workers AI"}
                </b>
              </span>
              <span>
                Model <b>{modelLabel}</b>
              </span>
              <span>
                Chế độ{" "}
                <b>{MODES.find((item) => item.id === settings.mode)?.label}</b>
              </span>
              <span>
                Kích thước{" "}
                <b>
                  {dimensions.width} × {dimensions.height}
                </b>
              </span>
            </div>
          </div>
          <div className="inspector-block">
            <div className="inspector-block-title">
              Bắt đầu nhanh <span>01 / 03</span>
            </div>
            <div className="quick-steps">
              <div>
                <span className="step-badge done">1</span>
                <p>
                  <b>Nhập ý tưởng</b>
                  <small>Mô tả hình ảnh mong muốn</small>
                </p>
              </div>
              <div>
                <span className="step-badge">2</span>
                <p>
                  <b>Chỉnh thông số</b>
                  <small>Model, khung hình, độ chi tiết</small>
                </p>
              </div>
              <div>
                <span className="step-badge">3</span>
                <p>
                  <b>Tạo & tải về</b>
                  <small>Lưu ảnh trong bộ sưu tập</small>
                </p>
              </div>
            </div>
          </div>
          <div className="inspector-block resource-block">
            <div className="inspector-block-title">
              Về tài nguyên <BadgeCheck size={16} />
            </div>
            <p>
              WAI v17 và 2 LoRA trong notebook có phiên bản SHA-256 đối chiếu từ
              Civitai. <strong>Workers AI dùng model riêng</strong>, không chạy
              các trọng số đó.
            </p>
            <a href={COLAB_URL} target="_blank" rel="noopener noreferrer">
              Mở notebook WAI <ArrowUpRight size={15} />
            </a>
          </div>
          <div className="inspector-tip">
            <span>✦ MẸO NHỎ</span>
            <p>
              Muốn sửa lỗi tay hoặc mắt? Chọn <strong>Sửa vùng ảnh</strong>, tải
              ảnh lên rồi tô đúng chỗ cần chỉnh.
            </p>
          </div>
          <div className="inspector-foot">
            ✦ &nbsp; Make more. Imagine more.
          </div>
        </aside>
      </div>

      {toast && (
        <div className={`toast toast-${toast.type}`} role="status">
          <span>
            {toast.type === "success" ? (
              <Check size={16} />
            ) : (
              <Info size={16} />
            )}
          </span>
          {toast.message}
          <button type="button" title="Đóng" onClick={() => setToast(null)}>
            <X size={14} />
          </button>
        </div>
      )}

      {authOpen && (
        <div
          className="modal-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setAuthOpen(false);
          }}
        >
          <div
            className="dialog access-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="Mã truy cập"
          >
            <button
              type="button"
              className="dialog-close"
              onClick={() => setAuthOpen(false)}
            >
              <X size={19} />
            </button>
            <span className="dialog-icon">
              <LockKeyhole size={24} />
            </span>
            <h2>Mã truy cập studio</h2>
            <p>
              Đây là mã <strong>APP_ACCESS_TOKEN</strong> do bạn đặt cho ứng
              dụng bằng <code>wrangler secret put</code> — không phải Cloudflare
              API token. Chỉ lưu trong phiên trình duyệt này.
            </p>
            <label>
              MÃ TRUY CẬP
              <input
                autoFocus
                type="password"
                placeholder="Nhập mã bảo vệ ứng dụng"
                value={authInput}
                onChange={(event) => setAuthInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") saveAccessCode();
                }}
              />
            </label>
            <button
              type="button"
              className="dialog-primary"
              disabled={!authInput.trim()}
              onClick={saveAccessCode}
            >
              Lưu mã truy cập <ArrowRight size={16} />
            </button>
            <small>
              Không nhập mật khẩu Cloudflare hoặc token tài khoản vào đây.
            </small>
          </div>
        </div>
      )}

      {helpOpen && (
        <div
          className="modal-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setHelpOpen(false);
          }}
        >
          <div
            className="dialog help-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="Hướng dẫn"
          >
            <button
              type="button"
              className="dialog-close"
              onClick={() => setHelpOpen(false)}
            >
              <X size={19} />
            </button>
            <span className="dialog-icon">
              <BookOpen size={23} />
            </span>
            <h2>Cách sử dụng Mirai Studio</h2>
            <div className="help-list">
              <div>
                <span>01</span>
                <div>
                  <b>Triển khai trên Cloudflare</b>
                  <p>
                    Trong thư mục <code>web</code>, chạy{" "}
                    <code>npm install</code>, đặt secret{" "}
                    <code>APP_ACCESS_TOKEN</code> bằng Wrangler rồi{" "}
                    <code>npm run deploy</code>. Cloudflare AI binding có sẵn
                    trong cấu hình.
                  </p>
                </div>
              </div>
              <div>
                <span>02</span>
                <div>
                  <b>Nhập mã & tạo ảnh</b>
                  <p>
                    Chỉ dùng mã truy cập ứng dụng bạn tự đặt, chọn Cloudflare AI
                    và mô hình SDXL. Prompt, kích thước, steps, CFG, seed và số
                    ảnh được chuyển tới API thật.
                  </p>
                </div>
              </div>
              <div>
                <span>03</span>
                <div>
                  <b>Chỉnh ảnh hiện có</b>
                  <p>
                    Ảnh → ảnh dùng ảnh tham chiếu. Sửa vùng ảnh dùng cọ hồng
                    hoặc tải mask PNG trắng/đen; ảnh nguồn và mask được gửi lên
                    model để xử lý vùng đã chọn.
                  </p>
                </div>
              </div>
              <div>
                <span>04</span>
                <div>
                  <b>Muốn đúng WAI v17?</b>
                  <p>
                    Cloudflare Worker không tải checkpoint 6,94 GB. Cần máy GPU
                    riêng hỗ trợ API <code>/generate</code> và đặt hai secret{" "}
                    <code>GPU_BACKEND_URL</code>, <code>GPU_BACKEND_TOKEN</code>
                    . Notebook Colab hiện vẫn hoạt động độc lập.
                  </p>
                </div>
              </div>
            </div>
            <a
              className="help-colab"
              href={COLAB_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              Mở notebook WAI-illustrious <ExternalLink size={16} />
            </a>
          </div>
        </div>
      )}

      {lightbox && active && (
        <div
          className="lightbox-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setLightbox(false);
          }}
        >
          <div
            className="lightbox"
            role="dialog"
            aria-modal="true"
            aria-label="Chi tiết ảnh"
          >
            <button
              type="button"
              className="lightbox-close"
              onClick={() => setLightbox(false)}
              aria-label="Đóng"
            >
              <X size={22} />
            </button>
            <div className="lightbox-image">
              <img src={active.url} alt="Ảnh kết quả phóng to" />
            </div>
            <div className="lightbox-details">
              <span className="eyebrow">CHI TIẾT TÁC PHẨM</span>
              <h2>
                Khoảnh khắc
                <br />
                bạn vừa tạo.
              </h2>
              <div className="details-meta">
                <div>
                  <span>MODEL</span>
                  <b>{MODEL_INFO[active.model]?.name || active.model}</b>
                </div>
                <div>
                  <span>SEED</span>
                  <b>{active.seed}</b>
                </div>
                <div>
                  <span>KÍCH THƯỚC</span>
                  <b>
                    {active.width} × {active.height}
                  </b>
                </div>
                <div>
                  <span>THÔNG SỐ</span>
                  <b>
                    {active.steps} bước · CFG {active.cfg}
                  </b>
                </div>
                <div>
                  <span>THỜI GIAN</span>
                  <b>{formatTime(active.duration)}</b>
                </div>
              </div>
              <div className="prompt-detail">
                <span>PROMPT</span>
                <p>{active.prompt}</p>
              </div>
              <div className="lightbox-actions">
                <button
                  type="button"
                  className="primary"
                  onClick={() =>
                    downloadBlob(active.blob, saveFilename(active))
                  }
                >
                  <Download size={16} /> Tải xuống
                </button>
                <button type="button" onClick={() => useRecordAsSource(active)}>
                  <Brush size={16} /> Sửa ảnh
                </button>
                <button
                  type="button"
                  onClick={() => {
                    copyPrompt(active.prompt);
                  }}
                >
                  <Copy size={16} /> Prompt
                </button>
                <button type="button" onClick={() => toggleFavorite(active)}>
                  <Heart
                    size={16}
                    fill={active.favorite ? "currentColor" : "none"}
                  />
                </button>
                <button type="button" onClick={() => removeRecord(active)}>
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
