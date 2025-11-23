# RAGDao

一本基于 GitBook 构建的开源书项目。

## 项目结构（核心路径）

- 项目根目录：本仓库
- GitBook root 路径：`.`

在 GitBook root 下，你会看到：

- `README.md`：作为书的“前言 / 首页”
- `SUMMARY.md`：书的目录结构
- `docs/`：章节内容
- `assets/`：图片与资源

## 快速开始（本地 GitBook CLI）

1. 安装 GitBook CLI（经典版本）：

```bash
npm install -g gitbook-cli
```

2. 安装插件并本地预览（在项目根目录执行）：

```bash
gitbook install .
gitbook serve .
```

3. 构建静态网站：

```bash
gitbook build . _book
```

构建结果将输出到 `_book/` 目录，可部署到 GitHub Pages、Netlify、Vercel 或任意静态服务器。

## GitBook 云端集成（可选）

- 在 gitbook.com 创建一个 Space
- 绑定本仓库与指定分支（如 `main`）
- 将 GitBook root 配置为 `.`，即可自动识别 README 与 SUMMARY

## 协议

本书在**非商业用途**下可免费使用和修改，任何商业使用需要获得作者书面授权。
详情见 [`LICENSE`](LICENSE)。
