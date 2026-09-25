export const MODEL_INFO = {
  "sdxl-base": {
    name: "SDXL Base 1.0",
    badge: "Cloudflare",
    description: "Cân bằng chi tiết và khả năng bám prompt.",
    suggestedSteps: 20,
  },
  "sdxl-lightning": {
    name: "SDXL Lightning",
    badge: "Nhanh",
    description: "Tạo nhanh trên Workers AI, thường hợp 4–8 bước.",
    suggestedSteps: 6,
  },
  "wai-v17": {
    name: "WAI-illustrious v17",
    badge: "GPU riêng",
    description: "Checkpoint anime gốc + LoRA; cần máy GPU kết nối.",
    suggestedSteps: 25,
  },
};

export const SIZE_PRESETS = [
  {
    id: "square",
    label: "Vuông",
    ratio: "1:1",
    width: 1024,
    height: 1024,
    icon: "square",
  },
  {
    id: "portrait",
    label: "Chân dung",
    ratio: "2:3",
    width: 832,
    height: 1216,
    icon: "portrait",
  },
  {
    id: "landscape",
    label: "Phong cảnh",
    ratio: "3:2",
    width: 1216,
    height: 832,
    icon: "landscape",
  },
];

export const STYLES = [
  {
    name: "Anime cổ điển",
    tag: "classic anime illustration, clean lineart, vibrant colors",
  },
  {
    name: "Cinematic",
    tag: "cinematic anime film still, dramatic lighting, depth of field",
  },
  {
    name: "Màu nước",
    tag: "soft watercolor anime painting, delicate brushwork",
  },
  {
    name: "Ánh sáng neon",
    tag: "neon lights, cyberpunk anime, glowing reflections",
  },
  {
    name: "Bầu trời mơ",
    tag: "dreamy pastel sky, atmospheric anime background",
  },
];

export const INSPIRATION = [
  {
    title: "Đêm hoa anh đào",
    subtitle: "Chân dung · lãng mạn",
    image: "/samples/sakura-night.jpg",
    prompt:
      "adult woman, dark blue hair, holding a transparent umbrella beneath cherry blossoms at dusk, petals in the air, traditional Japanese street, cinematic anime illustration, detailed eyes, gentle expression, natural hands, soft lavender light, masterpiece, best quality",
  },
  {
    title: "Chân trời mùa hạ",
    subtitle: "Phong cảnh · phiêu lưu",
    image: "/samples/sky-garden.jpg",
    prompt:
      "small traveler overlooking a sunlit coastal village from a hillside garden, sea shimmering at dawn, enormous pastel clouds, lush foliage, expansive anime background painting, rich atmosphere, masterpiece, best quality",
  },
  {
    title: "Phố trong mưa",
    subtitle: "Nhân vật · cyberpunk",
    image: "/samples/neon-runner.jpg",
    prompt:
      "adult man on a rooftop above a futuristic rainy city at night, indigo jacket, cyan and coral neon reflections, cinematic anime key visual, dynamic composition, expressive eyes, masterpiece, best quality",
  },
];

export const DEFAULTS = {
  provider: "cloudflare",
  model: "sdxl-base",
  mode: "txt2img",
  prompt:
    "adult anime girl, long dark hair, gentle smile, cherry blossoms falling in the wind, soft sunset light, elegant detailed eyes, natural hands, cinematic anime illustration, masterpiece, best quality",
  negative_prompt:
    "low quality, blurry, bad anatomy, extra fingers, extra limbs, deformed hands, uneven eyes, text, watermark",
  width: 1024,
  height: 1024,
  steps: 20,
  cfg: 6,
  seed: -1,
  count: 1,
  strength: 0.45,
  scheduler: "euler_a",
  clip_skip: 2,
  loras: {
    anatomy: { enabled: true, weight: 0.55 },
    eyes: { enabled: true, weight: 0.45 },
  },
};

export const COLAB_URL =
  "https://colab.research.google.com/github/manhlee1196-boop/ai-anime/blob/arena/01a0d84b-ai-anime/WAI_Illustrious_Colab.ipynb";
