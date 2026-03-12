// vite.config.js
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: '0.0.0.0',  // 监听所有网络接口
    port: 5174,
    strictPort: true  // 端口被占用时不再自动切换
  }
})
