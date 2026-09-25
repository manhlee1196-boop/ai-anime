// One Cloudflare Worker serves the Vite assets and runs the generation API.
// IMPORTANT: Workers AI has its own SDXL models; it does not host WAI v17.
const CF_MODELS = Object.freeze({
  "sdxl-base": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
  "sdxl-lightning": "@cf/bytedance/stable-diffusion-xl-lightning",
});
const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff",
};
const MAX_BODY_BYTES = 5_000_000;
const MAX_IMAGE_BYTES = 2_000_000;
const MAX_MASK_BYTES = 1_000_000;

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: JSON_HEADERS });
}

function numeric(value, name, min, max, integer = false) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < min ||
    value > max ||
    (integer && !Number.isInteger(value))
  ) {
    throw new ApiError(
      400,
      `${name} phải là ${integer ? "số nguyên" : "số"} từ ${min} đến ${max}.`,
    );
  }
  return value;
}

function imageBytes(encoded, field, maxBytes, mask = false) {
  if (
    typeof encoded !== "string" ||
    !encoded ||
    encoded.length > maxBytes * 1.38 + 100
  ) {
    throw new ApiError(
      400,
      `${field} thiếu hoặc vượt quá dung lượng cho phép.`,
    );
  }
  const source = encoded.replace(/^data:image\/(?:png|jpeg|webp);base64,/i, "");
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(source))
    throw new ApiError(400, `${field} không phải ảnh base64 hợp lệ.`);
  let binary;
  try {
    binary = atob(source);
  } catch {
    throw new ApiError(400, `${field} không phải ảnh base64 hợp lệ.`);
  }
  if (!binary.length || binary.length > maxBytes)
    throw new ApiError(400, `${field} vượt quá dung lượng cho phép.`);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  const png =
    bytes.length >= 8 &&
    bytes[0] === 137 &&
    bytes[1] === 80 &&
    bytes[2] === 78 &&
    bytes[3] === 71 &&
    bytes[4] === 13 &&
    bytes[5] === 10 &&
    bytes[6] === 26 &&
    bytes[7] === 10;
  const jpeg =
    bytes.length >= 3 &&
    bytes[0] === 255 &&
    bytes[1] === 216 &&
    bytes[2] === 255;
  const webp =
    bytes.length >= 12 &&
    String.fromCharCode(...bytes.slice(0, 4)) === "RIFF" &&
    String.fromCharCode(...bytes.slice(8, 12)) === "WEBP";
  if (mask ? !png : !(png || jpeg || webp))
    throw new ApiError(
      400,
      `${field} phải là ảnh ${mask ? "PNG" : "PNG, JPEG hoặc WebP"}.`,
    );
  return { bytes, base64: source };
}

function loraOption(value, name) {
  if (!value || typeof value !== "object" || typeof value.enabled !== "boolean")
    throw new ApiError(400, `Thiết lập ${name} không hợp lệ.`);
  return {
    enabled: value.enabled,
    weight: numeric(value.weight, `Cường độ ${name}`, 0.1, 1),
  };
}

export function validatePayload(data) {
  if (!data || typeof data !== "object" || Array.isArray(data))
    throw new ApiError(400, "Thiết lập ảnh không hợp lệ.");
  const provider = data.provider;
  if (provider !== "cloudflare" && provider !== "wai")
    throw new ApiError(400, "Nguồn model không hợp lệ.");
  const model = data.model;
  if (
    typeof model !== "string" ||
    (provider === "cloudflare"
      ? !Object.hasOwn(CF_MODELS, model)
      : model !== "wai-v17")
  )
    throw new ApiError(400, "Model không được hỗ trợ.");
  const mode = data.mode;
  if (!["txt2img", "img2img", "inpaint"].includes(mode))
    throw new ApiError(400, "Chế độ tạo ảnh không hợp lệ.");
  const prompt = typeof data.prompt === "string" ? data.prompt.trim() : "";
  if (!prompt || prompt.length > 2000)
    throw new ApiError(400, "Prompt phải có từ 1 đến 2000 ký tự.");
  const negative_prompt =
    typeof data.negative_prompt === "string" ? data.negative_prompt.trim() : "";
  if (negative_prompt.length > 1500)
    throw new ApiError(400, "Negative prompt tối đa 1500 ký tự.");
  const width = numeric(data.width, "Chiều rộng", 512, 1344, true);
  const height = numeric(data.height, "Chiều cao", 512, 1344, true);
  if (
    width % 8 ||
    height % 8 ||
    width * height > 1_500_000 ||
    Math.max(width / height, height / width) > 1.75
  ) {
    throw new ApiError(
      400,
      "Kích thước phải chia hết cho 8, không vượt 1,5 MP hoặc tỷ lệ 1,75:1.",
    );
  }
  const steps = numeric(
    data.steps,
    "Steps",
    1,
    provider === "wai" ? 45 : 20,
    true,
  );
  const cfg = numeric(data.cfg, "CFG / Guidance", 1, 12);
  const seed =
    data.seed === -1
      ? crypto.getRandomValues(new Uint32Array(1))[0]
      : numeric(data.seed, "Seed", 0, 4294967295, true);
  const strength =
    mode === "txt2img"
      ? 1
      : numeric(data.strength, "Denoise strength", 0.2, 0.95);
  let image;
  let mask;
  if (mode !== "txt2img")
    image = imageBytes(data.image_b64, "Ảnh nguồn", MAX_IMAGE_BYTES);
  if (mode === "inpaint")
    mask = imageBytes(data.mask_b64, "Mask", MAX_MASK_BYTES, true);
  const result = {
    provider,
    model,
    mode,
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    seed,
    strength,
    image,
    mask,
  };
  if (provider === "wai") {
    result.loras = {
      anatomy: loraOption(data.loras?.anatomy, "Anatomy Helper"),
      eyes: loraOption(data.loras?.eyes, "Perfect Eyes"),
    };
    if (!["euler_a", "dpmpp_2m"].includes(data.scheduler))
      throw new ApiError(400, "Sampler không được hỗ trợ.");
    result.scheduler = data.scheduler;
    result.clip_skip = numeric(data.clip_skip, "Clip skip", 1, 2, true);
  }
  return result;
}

export function publicConfig(env) {
  return {
    cloudflare: Boolean(env.AI),
    wai: Boolean(env.GPU_BACKEND_URL && env.GPU_BACKEND_TOKEN),
    authRequired: true,
    configured: Boolean(env.APP_ACCESS_TOKEN),
    localPreview: false,
    models: CF_MODELS,
  };
}

function safeEquals(left, right) {
  if (typeof left !== "string" || typeof right !== "string") return false;
  // Compare the entire padded input rather than returning at the first mismatch.
  let mismatch = left.length ^ right.length;
  const count = Math.max(left.length, right.length);
  for (let i = 0; i < count; i++)
    mismatch |= (left.charCodeAt(i) || 0) ^ (right.charCodeAt(i) || 0);
  return mismatch === 0;
}

async function readLimited(request) {
  if (!request.body) return "";
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY_BYTES)
        throw new ApiError(413, "Dữ liệu ảnh quá lớn (tối đa 5 MB).");
      text += decoder.decode(value, { stream: true });
    }
    return text + decoder.decode();
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
}

async function runCloudflare(env, input) {
  if (!env.AI)
    throw new ApiError(
      503,
      "Workers AI chưa được cấu hình trong Cloudflare Worker.",
    );
  const payload = {
    prompt: input.prompt,
    negative_prompt: input.negative_prompt,
    width: input.width,
    height: input.height,
    num_steps: input.steps,
    guidance: input.cfg,
    seed: input.seed,
  };
  if (input.mode !== "txt2img") {
    payload.strength = input.strength;
    if (input.mode === "inpaint") {
      // The Cloudflare SDXL schema expects byte arrays for image+mask in inpaint mode.
      payload.image = Array.from(input.image.bytes);
      payload.mask = Array.from(input.mask.bytes);
    } else {
      payload.image_b64 = input.image.base64;
    }
  }
  const result = await env.AI.run(CF_MODELS[input.model], payload);
  if (result instanceof Response) {
    if (!result.ok)
      throw new ApiError(502, `Workers AI trả về lỗi ${result.status}.`);
    const type = result.headers
      .get("Content-Type")
      ?.split(";")[0]
      .toLowerCase()
      .trim();
    if (type && type !== "image/png" && type !== "application/octet-stream")
      throw new ApiError(502, "Workers AI không trả về ảnh PNG.");
    return result;
  }
  if (
    result instanceof ReadableStream ||
    result instanceof ArrayBuffer ||
    ArrayBuffer.isView(result)
  ) {
    return new Response(result, { headers: { "Content-Type": "image/png" } });
  }
  throw new Error("Invalid Workers AI output");
}

async function runWai(env, input) {
  if (!env.GPU_BACKEND_URL || !env.GPU_BACKEND_TOKEN)
    throw new ApiError(
      503,
      "WAI v17 cần máy GPU riêng. Hãy cấu hình GPU_BACKEND_URL và GPU_BACKEND_TOKEN.",
    );
  let backend;
  try {
    backend = new URL(env.GPU_BACKEND_URL);
  } catch {
    throw new ApiError(503, "GPU_BACKEND_URL không hợp lệ.");
  }
  if (backend.protocol !== "https:" || backend.username || backend.password)
    throw new ApiError(
      503,
      "GPU_BACKEND_URL phải dùng HTTPS và không chứa mật khẩu trong URL.",
    );
  backend.pathname = backend.pathname.replace(/\/$/, "") + "/generate";
  const payload = {
    mode: input.mode,
    model: input.model,
    prompt: input.prompt,
    negative_prompt: input.negative_prompt,
    width: input.width,
    height: input.height,
    steps: input.steps,
    cfg: input.cfg,
    seed: input.seed,
    strength: input.strength,
    scheduler: input.scheduler,
    clip_skip: input.clip_skip,
    loras: input.loras,
    ...(input.image ? { image_b64: input.image.base64 } : {}),
    ...(input.mask ? { mask_b64: input.mask.base64 } : {}),
  };
  const res = await fetch(backend, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.GPU_BACKEND_TOKEN}`,
    },
    body: JSON.stringify(payload),
    redirect: "error",
  });
  if (!res.ok)
    throw new ApiError(
      502,
      `Máy GPU trả về lỗi ${res.status}. Kiểm tra backend và nhật ký máy GPU.`,
    );
  const mime = res.headers.get("content-type")?.split(";")[0] || "";
  if (!["image/png", "image/jpeg", "image/webp"].includes(mime))
    throw new ApiError(502, "Máy GPU không trả về ảnh hợp lệ.");
  return res;
}

export async function handleRequest(request, env) {
  const url = new URL(request.url);
  if (url.pathname === "/api/config" && request.method === "GET")
    return json(publicConfig(env));
  if (url.pathname !== "/api/generate") {
    if (url.pathname.startsWith("/api/"))
      return json({ error: "Đường dẫn API không tồn tại." }, 404);
    return env.ASSETS
      ? env.ASSETS.fetch(request)
      : json({ error: "Không tìm thấy giao diện." }, 404);
  }
  if (request.method !== "POST")
    return json({ error: "Chỉ hỗ trợ POST." }, 405);
  const origin = request.headers.get("Origin");
  if (origin && origin !== url.origin)
    return json({ error: "Yêu cầu khác nguồn không được phép." }, 403);
  if (!env.APP_ACCESS_TOKEN)
    return json(
      { error: "Chưa thiết lập APP_ACCESS_TOKEN. Xem hướng dẫn triển khai." },
      503,
    );
  const bearer = /^Bearer (.+)$/i.exec(
    request.headers.get("Authorization") || "",
  );
  if (!safeEquals(bearer?.[1], env.APP_ACCESS_TOKEN)) {
    return json(
      { error: "Mã truy cập chưa đúng. Kiểm tra mã của bạn rồi thử lại." },
      401,
    );
  }
  if (
    !request.headers
      .get("content-type")
      ?.toLowerCase()
      .startsWith("application/json")
  )
    return json({ error: "Gửi dữ liệu dạng JSON." }, 415);
  if (Number(request.headers.get("content-length") || 0) > MAX_BODY_BYTES)
    return json({ error: "Dữ liệu ảnh quá lớn (tối đa 5 MB)." }, 413);
  try {
    const text = await readLimited(request);
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new ApiError(400, "JSON không hợp lệ.");
    }
    const input = validatePayload(data);
    const start = Date.now();
    const imageResponse =
      input.provider === "wai"
        ? await runWai(env, input)
        : await runCloudflare(env, input);
    const headers = new Headers(imageResponse.headers);
    headers.set("Cache-Control", "no-store");
    headers.set("X-Content-Type-Options", "nosniff");
    headers.set("X-Image-Seed", String(input.seed));
    headers.set("X-Image-Model", input.model);
    headers.set("X-Image-Provider", input.provider);
    headers.set("X-Generation-Ms", String(Date.now() - start));
    // The hosted SDXL models' documented binary output is PNG. Some runtime
    // adapters use application/octet-stream for the same binary response.
    if (input.provider === "cloudflare" || !headers.has("Content-Type"))
      headers.set("Content-Type", "image/png");
    // Do not expose any backend authentication / upstream headers to the browser.
    for (const key of [...headers.keys()]) {
      if (
        ![
          "content-type",
          "cache-control",
          "x-content-type-options",
          "x-image-seed",
          "x-image-model",
          "x-image-provider",
          "x-generation-ms",
        ].includes(key.toLowerCase())
      )
        headers.delete(key);
    }
    return new Response(imageResponse.body, { status: 200, headers });
  } catch (error) {
    if (error instanceof ApiError)
      return json({ error: error.message }, error.status);
    // Avoid leaking prompt, access secrets, upstream URLs or stack traces.
    console.error(
      "Image generation failed:",
      error instanceof Error ? error.name : "UnknownError",
    );
    return json(
      {
        error:
          "Không tạo được ảnh. Kiểm tra AI binding, hạn mức Workers AI hoặc trạng thái máy GPU.",
      },
      502,
    );
  }
}

export default { fetch: handleRequest };
