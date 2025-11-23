# 如何贡献本书

感谢你愿意为本书做出贡献！

## 1. 可以做什么？

- 修正文中的错别字、语病、表达不清
- 提出勘误或改进建议
- 补充示例、插图、参考资料
- 新增或重构章节内容
- 优化 GitBook 结构或插件配置

## 2. 基本流程

1. Fork 本仓库
2. 从 `main` 创建新分支，例如 `feature/chapter-4-improvement`
3. 在 GitBook root 下的 `docs/` 目录中编辑或新增相应章节
4. 本地预览（假设 GitBook root 为当前目录）：

   ```bash
   gitbook install
   gitbook serve
   ```

   若 GitBook root 在子目录（如 `book/`），请相应进入该目录或指定路径。

5. 提交更改并发起 Pull Request
   - 请在 PR 中简要说明本次修改的内容和动机

## 3. 写作规范（建议）

- 使用 Markdown 编写
- 标题层级从 `#`（章节标题）开始，依次向下
- 中英文之间适当留空格，例如：`Python 库`、`AI 产品`
- 术语尽量保持统一（可在后续增加术语表）
- 代码块请注明语言高亮，如：

  ```python
  def example():
      print("Hello, open book!")
  ```

## 4. Issue 建议

- 勘误：请附上章节号和原文片段
- 新内容建议：简要说明希望增加的主题和价值
- 若是较大的结构调整建议，可以先在 Discussion / Issue 中讨论
