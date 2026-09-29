import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Vite preview is deliberately offline: it previews the actual interface,
// while generation requires deploying the Worker with a Cloudflare AI binding.
function localApiNotice() {
  return {
    name: "local-api-notice",
    configureServer(server) {
      server.middlewares.use("/api", (req, res) => {
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        if (req.method === "GET" && req.url?.startsWith("/config")) {
          res.end(
            JSON.stringify({
              cloudflare: false,
              wai: false,
              authRequired: true,
              localPreview: true,
              message:
                "Bản xem trước giao diện. Deploy Worker để gọi Workers AI.",
            }),
          );
          return;
        }
        res.statusCode = 503;
        res.end(
          JSON.stringify({
            error:
              "Đây là bản xem trước giao diện. Hãy triển khai lên Cloudflare để tạo ảnh thật.",
          }),
        );
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), localApiNotice()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    allowedHosts: true,
  },
});
