import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs separately (`promptloop serve`), so proxy /api to it.
// In production FastAPI serves this build itself, so the frontend uses relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/docs": "http://127.0.0.1:8000",
      "/openapi.json": "http://127.0.0.1:8000",
    },
  },
});
