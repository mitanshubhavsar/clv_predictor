import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Set VITE_BASE to the repository name when publishing, e.g. "/my-repo/".
export default defineConfig(({ command }) => ({
  base: command === "build" ? process.env.VITE_BASE ?? "/clv-stepwise/" : "/",
  plugins: [react()],
}));
