import React, { useEffect, useRef, useState } from "react";
import { Brush, Eraser, RotateCcw, Trash2, Upload } from "lucide-react";

export default function MaskEditor({ source, onChange, onError }) {
  const maskRef = useRef(null);
  const overlayRef = useRef(null);
  const drawing = useRef(false);
  const last = useRef(null);
  const undoStack = useRef([]);
  const [tool, setTool] = useState("brush");
  const [size, setSize] = useState(38);
  const [canUndo, setCanUndo] = useState(false);
  const [painted, setPainted] = useState(false);

  function updatePreview() {
    const mask = maskRef.current;
    const overlay = overlayRef.current;
    if (!mask || !overlay) return;
    const ctx = mask.getContext("2d");
    const image = ctx.getImageData(0, 0, mask.width, mask.height);
    const preview = overlay.getContext("2d");
    const tinted = preview.createImageData(mask.width, mask.height);
    let any = false;
    for (let i = 0; i < image.data.length; i += 4) {
      if (image.data[i] > 127) {
        tinted.data[i] = 245;
        tinted.data[i + 1] = 126;
        tinted.data[i + 2] = 186;
        tinted.data[i + 3] = 165;
        any = true;
      }
    }
    preview.putImageData(tinted, 0, 0);
    setPainted(any);
    onChange(any ? mask.toDataURL("image/png") : null);
  }

  useEffect(() => {
    if (!source) return;
    for (const canvas of [maskRef.current, overlayRef.current]) {
      canvas.width = source.width;
      canvas.height = source.height;
    }
    const ctx = maskRef.current.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, source.width, source.height);
    overlayRef.current
      .getContext("2d")
      .clearRect(0, 0, source.width, source.height);
    undoStack.current = [];
    setCanUndo(false);
    setPainted(false);
    onChange(null);
  }, [source?.dataURL]); // A new image always starts with an empty mask.

  function position(event) {
    const rect = overlayRef.current.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) * source.width) / rect.width,
      y: ((event.clientY - rect.top) * source.height) / rect.height,
    };
  }

  function stroke(from, to) {
    const ctx = maskRef.current.getContext("2d");
    ctx.strokeStyle = tool === "brush" ? "#fff" : "#000";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = size;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(to.x, to.y, size / 2, 0, Math.PI * 2);
    ctx.fill();
    const preview = overlayRef.current.getContext("2d");
    preview.globalCompositeOperation =
      tool === "erase" ? "destination-out" : "source-over";
    preview.strokeStyle =
      tool === "erase" ? "#000" : "rgba(245, 126, 186, 0.68)";
    preview.fillStyle = preview.strokeStyle;
    preview.lineCap = "round";
    preview.lineWidth = size;
    preview.beginPath();
    preview.moveTo(from.x, from.y);
    preview.lineTo(to.x, to.y);
    preview.stroke();
    preview.beginPath();
    preview.arc(to.x, to.y, size / 2, 0, Math.PI * 2);
    preview.fill();
    preview.globalCompositeOperation = "source-over";
  }

  function start(event) {
    event.preventDefault();
    overlayRef.current.setPointerCapture(event.pointerId);
    const ctx = maskRef.current.getContext("2d");
    undoStack.current.push(ctx.getImageData(0, 0, source.width, source.height));
    if (undoStack.current.length > 8) undoStack.current.shift();
    setCanUndo(true);
    drawing.current = true;
    last.current = position(event);
    stroke(last.current, last.current);
  }

  function move(event) {
    if (!drawing.current) return;
    event.preventDefault();
    const point = position(event);
    stroke(last.current, point);
    last.current = point;
  }

  function stop() {
    if (drawing.current) updatePreview();
    drawing.current = false;
    last.current = null;
  }

  function undo() {
    const previous = undoStack.current.pop();
    if (!previous) return;
    maskRef.current.getContext("2d").putImageData(previous, 0, 0);
    setCanUndo(undoStack.current.length > 0);
    updatePreview();
  }

  function clear() {
    const ctx = maskRef.current.getContext("2d");
    undoStack.current.push(ctx.getImageData(0, 0, source.width, source.height));
    if (undoStack.current.length > 8) undoStack.current.shift();
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, source.width, source.height);
    setCanUndo(true);
    updatePreview();
  }

  async function importMask(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (file.type !== "image/png") {
      onError("Mask cần là ảnh PNG trắng/đen.");
      return;
    }
    try {
      const bitmap = await createImageBitmap(file);
      try {
        if (bitmap.width !== source.width || bitmap.height !== source.height) {
          throw new Error(
            `Mask phải có kích thước ${source.width} × ${source.height} px.`,
          );
        }
        const ctx = maskRef.current.getContext("2d");
        undoStack.current.push(
          ctx.getImageData(0, 0, source.width, source.height),
        );
        if (undoStack.current.length > 8) undoStack.current.shift();
        ctx.drawImage(bitmap, 0, 0);
        const pixels = ctx.getImageData(0, 0, source.width, source.height);
        for (let i = 0; i < pixels.data.length; i += 4) {
          const value =
            pixels.data[i] * 0.2126 +
            pixels.data[i + 1] * 0.7152 +
            pixels.data[i + 2] * 0.0722;
          pixels.data[i] =
            pixels.data[i + 1] =
            pixels.data[i + 2] =
              value >= 128 ? 255 : 0;
          pixels.data[i + 3] = 255;
        }
        ctx.putImageData(pixels, 0, 0);
        setCanUndo(true);
        updatePreview();
      } finally {
        bitmap.close();
      }
    } catch (error) {
      onError(error.message || "Không đọc được mask.");
    }
  }

  return (
    <div className="mask-editor">
      <div className="mask-toolbar">
        <div className="mask-tools-left">
          <button
            type="button"
            className={`mask-tool ${tool === "brush" ? "selected" : ""}`}
            onClick={() => setTool("brush")}
            title="Tô vùng cần sửa"
          >
            <Brush size={16} /> Tô mask
          </button>
          <button
            type="button"
            className={`mask-tool ${tool === "erase" ? "selected" : ""}`}
            onClick={() => setTool("erase")}
            title="Xóa vùng đã tô"
          >
            <Eraser size={16} /> Tẩy
          </button>
          <div className="brush-size">
            <span>Độ rộng</span>
            <input
              aria-label="Độ rộng cọ"
              type="range"
              min="8"
              max="120"
              value={size}
              onChange={(event) => setSize(Number(event.target.value))}
            />
            <b>{size}</b>
          </div>
        </div>
        <div className="mask-tools-right">
          <label
            className="icon-tool import-mask"
            title="Nhập mask PNG trắng/đen"
          >
            <Upload size={16} />
            <span>Nạp mask PNG</span>
            <input
              type="file"
              accept="image/png"
              onChange={importMask}
              hidden
            />
          </label>
          <button
            type="button"
            className="icon-tool"
            title="Hoàn tác"
            onClick={undo}
            disabled={!canUndo}
          >
            <RotateCcw size={17} />
          </button>
          <button
            type="button"
            className="icon-tool"
            title="Xóa mask"
            onClick={clear}
            disabled={!painted}
          >
            <Trash2 size={17} />
          </button>
        </div>
      </div>
      <div className="paint-wrap">
        <div
          className="paint-surface"
          style={{ aspectRatio: `${source.width} / ${source.height}` }}
        >
          <img
            src={source.dataURL}
            alt="Ảnh nguồn để sửa vùng"
            draggable="false"
          />
          <canvas
            ref={overlayRef}
            className="paint-overlay"
            aria-label="Vùng mask: tô các khu vực muốn sửa"
            onPointerDown={start}
            onPointerMove={move}
            onPointerUp={stop}
            onPointerCancel={stop}
          />
          <canvas ref={maskRef} className="hidden-canvas" aria-hidden="true" />
          <span className="paint-indicator">
            {painted
              ? "Vùng hồng = vùng được sửa"
              : "Tô lên vùng cần sửa: mắt, tay hoặc chân"}
          </span>
        </div>
      </div>
    </div>
  );
}
