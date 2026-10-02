import { readFile } from "node:fs/promises";

export const TAG_FILE = "danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv";
const source = new URL(`../../${TAG_FILE}`, import.meta.url);

// Keep one source of truth in the repository; publish it as a static asset.
export default function tagResource() {
  return {
    name: "tag-resource",
    configureServer(server) {
      server.middlewares.use(`/tags/${TAG_FILE}`, async (_req, res, next) => {
        try {
          res.setHeader("Content-Type", "text/csv; charset=utf-8");
          res.end(await readFile(source));
        } catch (error) {
          next(error);
        }
      });
    },
    async generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: `tags/${TAG_FILE}`,
        source: await readFile(source),
      });
    },
  };
}
