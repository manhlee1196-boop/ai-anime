export function getRandomSeed() {
  const bytes = new Uint32Array(1);
  crypto.getRandomValues(bytes);
  return bytes[0];
}

function canvasBlob(canvas, type, quality) {
  return new Promise((resolve) => canvas.toBlob(resolve, type, quality));
}

export async function prepareSource(file) {
  if (!["image/png", "image/jpeg", "image/webp"].includes(file?.type))
    throw new Error("Chọn ảnh PNG, JPEG hoặc WebP.");
  if (file.size > 12_000_000)
    throw new Error(
      "Ảnh nguồn tối đa 12 MB. Hãy thu nhỏ ảnh trước khi tải lên.",
    );
  const bitmap = await createImageBitmap(file);
  try {
    const longer = Math.max(bitmap.width, bitmap.height);
    const shorter = Math.min(bitmap.width, bitmap.height);
    if (!shorter || longer / shorter > 1.75)
      throw new Error(
        "Ảnh nguồn quá dài; tỷ lệ tối đa 1,75:1. Hãy cắt ảnh trước khi tải.",
      );
    // Upscale small images while preserving their aspect ratio. Always stay
    // within the 512px minimum and the 1024px maximum for efficient inference.
    const scale = Math.min(1024 / longer, Math.max(1, 512 / shorter));
    let width = Math.max(512, Math.round((bitmap.width * scale) / 8) * 8);
    let height = Math.max(512, Math.round((bitmap.height * scale) / 8) * 8);
    if (width / height > 1.75) height = Math.ceil(width / 1.75 / 8) * 8;
    if (height / width > 1.75) width = Math.ceil(height / 1.75 / 8) * 8;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);
    let blob = await canvasBlob(canvas, "image/png");
    if (!blob) throw new Error("Không đọc được ảnh nguồn.");
    if (blob.size > 1_900_000)
      blob = await canvasBlob(canvas, "image/jpeg", 0.86);
    if (!blob || blob.size > 1_900_000)
      throw new Error(
        "Ảnh đã nén vẫn quá lớn. Hãy thử ảnh đơn giản hoặc độ phân giải thấp hơn.",
      );
    const dataURL = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Không đọc được ảnh."));
      reader.readAsDataURL(blob);
    });
    return { dataURL, width, height, name: file.name || "Ảnh được chọn" };
  } finally {
    bitmap.close();
  }
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 120_000);
}

export function formatTime(milliseconds) {
  if (!milliseconds) return "—";
  return `${(milliseconds / 1000).toFixed(1)} giây`;
}
