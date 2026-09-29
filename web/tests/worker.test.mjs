import test from "node:test";
import assert from "node:assert/strict";
import {
  handleRequest,
  publicConfig,
  validatePayload,
} from "../worker/index.js";

const origin = "https://studio.example.test";
const png = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 0]);
const pngBase64 = Buffer.from(png).toString("base64");
const base = {
  provider: "cloudflare",
  model: "sdxl-base",
  mode: "txt2img",
  prompt: "anime landscape",
  negative_prompt: "blur",
  width: 1024,
  height: 1024,
  steps: 20,
  cfg: 6,
  seed: 42,
  strength: 0.45,
};

function request(body, headers = {}) {
  return new Request(`${origin}/api/generate`, {
    method: "POST",
    body: JSON.stringify(body),
    headers: {
      "Content-Type": "application/json",
      Origin: origin,
      Authorization: "Bearer studio-secret",
      ...headers,
    },
  });
}
function env(ai) {
  return { AI: ai, APP_ACCESS_TOKEN: "studio-secret" };
}

const stream = () =>
  new ReadableStream({
    start(controller) {
      controller.enqueue(png);
      controller.close();
    },
  });

async function error(response, status, pattern) {
  assert.equal(response.status, status);
  assert.match((await response.json()).error, pattern);
}

test("public capabilities omit access and GPU secrets", async () => {
  const bindings = {
    AI: { run() {} },
    APP_ACCESS_TOKEN: "private",
    GPU_BACKEND_URL: "https://gpu.internal",
    GPU_BACKEND_TOKEN: "secret",
  };
  const response = await handleRequest(
    new Request(`${origin}/api/config`),
    bindings,
  );
  assert.equal(response.status, 200);
  const info = await response.json();
  assert.equal(info.cloudflare, true);
  assert.equal(info.wai, true);
  assert.equal(info.configured, true);
  assert.doesNotMatch(JSON.stringify(info), /private|gpu\.internal|secret/);
  assert.deepEqual(publicConfig({}).models, info.models);
});

test("fail-closed access control, invalid origin and missing JSON", async () => {
  await error(
    await handleRequest(request(base), { AI: { run() {} } }),
    503,
    /APP_ACCESS_TOKEN/,
  );
  await error(
    await handleRequest(
      request(base, { Authorization: "Bearer wrong" }),
      env({ run() {} }),
    ),
    401,
    /mã truy cập/i,
  );
  await error(
    await handleRequest(
      request(base, { Authorization: "studio-secret" }),
      env({ run() {} }),
    ),
    401,
    /mã truy cập/i,
  );
  await error(
    await handleRequest(
      request(base, { Origin: "https://other.example" }),
      env({ run() {} }),
    ),
    403,
    /khác nguồn/i,
  );
  await error(
    await handleRequest(
      request(base, { "Content-Type": "text/plain" }),
      env({ run() {} }),
    ),
    415,
    /JSON/,
  );
  await error(
    await handleRequest(
      new Request(`${origin}/api/generate`),
      env({ run() {} }),
    ),
    405,
    /POST/,
  );
});

test("Cloudflare SDXL sends all supported parameters and returns real image bytes", async () => {
  let captured;
  const ai = {
    run: async (model, input) => {
      captured = { model, input };
      return stream();
    },
  };
  const res = await handleRequest(request(base), env(ai));
  assert.equal(res.status, 200);
  assert.deepEqual(
    Array.from(new Uint8Array(await res.arrayBuffer())),
    Array.from(png),
  );
  assert.equal(res.headers.get("Content-Type"), "image/png");
  assert.equal(res.headers.get("X-Image-Seed"), "42");
  assert.equal(res.headers.get("Cache-Control"), "no-store");
  assert.equal(captured.model, "@cf/stabilityai/stable-diffusion-xl-base-1.0");
  assert.deepEqual(captured.input, {
    prompt: base.prompt,
    negative_prompt: base.negative_prompt,
    width: 1024,
    height: 1024,
    num_steps: 20,
    guidance: 6,
    seed: 42,
  });
  assert.equal(JSON.stringify(captured.input).includes("lora"), false);
});

test("img2img uses source base64; inpaint uses PNG byte arrays for SDXL Lightning", async () => {
  const calls = [];
  const ai = {
    run: async (model, input) => {
      calls.push({ model, input });
      return stream();
    },
  };
  const image = `data:image/png;base64,${pngBase64}`;
  const img = await handleRequest(
    request({
      ...base,
      model: "sdxl-lightning",
      mode: "img2img",
      steps: 6,
      strength: 0.6,
      image_b64: image,
    }),
    env(ai),
  );
  assert.equal(img.status, 200);
  const paint = await handleRequest(
    request({ ...base, mode: "inpaint", image_b64: image, mask_b64: image }),
    env(ai),
  );
  assert.equal(paint.status, 200);
  assert.equal(calls[0].model, "@cf/bytedance/stable-diffusion-xl-lightning");
  assert.equal(calls[0].input.image_b64, pngBase64);
  assert.equal(calls[0].input.strength, 0.6);
  assert.deepEqual(calls[1].input.image, Array.from(png));
  assert.deepEqual(calls[1].input.mask, Array.from(png));
  assert.equal("image_b64" in calls[1].input, false);
});

test("rejects unsupported model, steps, size, malformed image and WAI LoRA on Cloudflare", async () => {
  const noOp = env({
    run() {
      throw new Error("must not be called");
    },
  });
  for (const [value, pattern] of [
    [{ ...base, model: "wai-v17" }, /Model/],
    [{ ...base, model: "constructor" }, /Model/],
    [{ ...base, steps: 25 }, /Steps/],
    [{ ...base, width: 1023 }, /Kích thước/],
    [{ ...base, width: 1344, height: 1344 }, /Kích thước/],
    [{ ...base, prompt: " " }, /Prompt/],
    [{ ...base, mode: "img2img" }, /Ảnh nguồn/],
    [{ ...base, mode: "inpaint", image_b64: pngBase64 }, /Mask/],
    [
      {
        ...base,
        mode: "inpaint",
        image_b64: pngBase64,
        mask_b64: Buffer.from("not a png").toString("base64"),
      },
      /Mask/,
    ],
  ]) {
    await error(await handleRequest(request(value), noOp), 400, pattern);
  }
});

test("WAI requires external GPU and does not run with Workers AI model", async () => {
  const wai = {
    ...base,
    provider: "wai",
    model: "wai-v17",
    steps: 25,
    scheduler: "euler_a",
    clip_skip: 2,
    loras: {
      anatomy: { enabled: true, weight: 0.55 },
      eyes: { enabled: true, weight: 0.45 },
    },
  };
  await error(
    await handleRequest(
      request(wai),
      env({
        run() {
          throw new Error("Workers AI must not run WAI");
        },
      }),
    ),
    503,
    /GPU/,
  );
  let call;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    call = { url: String(url), options };
    return new Response(png, {
      headers: { "Content-Type": "image/png", "X-Super-Secret": "not public" },
    });
  };
  try {
    const res = await handleRequest(request(wai), {
      ...env(null),
      GPU_BACKEND_URL: "https://gpu.example.org/api",
      GPU_BACKEND_TOKEN: "gpu-password",
    });
    assert.equal(res.status, 200);
    assert.equal(call.url, "https://gpu.example.org/api/generate");
    assert.equal(call.options.headers.Authorization, "Bearer gpu-password");
    assert.deepEqual(JSON.parse(call.options.body).loras, wai.loras);
    assert.equal(res.headers.has("X-Super-Secret"), false);
    assert.deepEqual(
      Array.from(new Uint8Array(await res.arrayBuffer())),
      Array.from(png),
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("rejects insecure GPU origin and invalid auth before any model call", async () => {
  const wai = {
    ...base,
    provider: "wai",
    model: "wai-v17",
    steps: 25,
    scheduler: "euler_a",
    clip_skip: 2,
    loras: {
      anatomy: { enabled: false, weight: 0.5 },
      eyes: { enabled: false, weight: 0.5 },
    },
  };
  await error(
    await handleRequest(request(wai), {
      ...env(null),
      GPU_BACKEND_URL: "http://127.0.0.1:9000",
      GPU_BACKEND_TOKEN: "private",
    }),
    503,
    /HTTPS/,
  );
  await error(
    await handleRequest(request(wai, { Authorization: "Bearer no" }), {
      ...env(null),
      GPU_BACKEND_URL: "https://gpu.example.org",
      GPU_BACKEND_TOKEN: "private",
    }),
    401,
    /mã truy cập/i,
  );
});

test("Workers AI error responses are not incorrectly presented as successful images", async () => {
  await error(
    await handleRequest(
      request(base),
      env({ run: async () => new Response("quota exceeded", { status: 429 }) }),
    ),
    502,
    /429/,
  );
  await error(
    await handleRequest(
      request(base),
      env({
        run: async () =>
          new Response('{"error":"invalid"}', {
            headers: { "Content-Type": "application/json" },
          }),
      }),
    ),
    502,
    /PNG/,
  );
  const binary = await handleRequest(
    request(base),
    env({
      run: async () =>
        new Response(png, {
          headers: { "Content-Type": "application/octet-stream" },
        }),
    }),
  );
  assert.equal(binary.status, 200);
  assert.equal(binary.headers.get("Content-Type"), "image/png");
});

test("request bodies are bounded even when Content-Length is absent or misleading", async () => {
  const tooLarge = new Request(`${origin}/api/generate`, {
    method: "POST",
    body: " ".repeat(5_000_100),
    headers: {
      Origin: origin,
      Authorization: "Bearer studio-secret",
      "Content-Type": "application/json",
    },
  });
  await error(await handleRequest(tooLarge, env({ run() {} })), 413, /5 MB/);
});

test("API path never turns into a public proxy", async () => {
  await error(
    await handleRequest(new Request(`${origin}/api/other`), env({})),
    404,
    /không tồn tại/,
  );
  assert.throws(
    () => validatePayload({ ...base, provider: "other" }),
    /Nguồn model/,
  );
});
