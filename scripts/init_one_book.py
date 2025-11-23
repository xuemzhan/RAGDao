#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 GitBook 的开源书项目初始化脚本。

功能：
- 创建项目目录与 GitBook 友好的结构
- 生成 SUMMARY.md、book.json 等关键文件
- LICENSE 为「非商业可用，商业使用需授权」
- 可指定 GitBook root 路径（相对项目根）

用法示例：
    # 根目录即为 GitBook root
    python init_gitbook_book.py my-open-book

    # 在 book/ 目录下作为 GitBook root
    python init_gitbook_book.py my-open-book -r book
"""

import argparse
from pathlib import Path
import sys
import textwrap


def create_file(path: Path, content: str):
    """如果文件不存在则创建，并写入内容。"""
    if path.exists():
        print(f"[跳过] 文件已存在: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[创建] {path}")


# ========== 文本模板部分 ==========

def generate_repo_readme(project_name: str, root_rel: str) -> str:
    return textwrap.dedent(f"""\
    # {project_name}

    一本基于 GitBook 构建的开源书项目。

    ## 项目结构（核心路径）

    - 项目根目录：本仓库
    - GitBook root 路径：`{root_rel or "."}`

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
    gitbook install {root_rel or "."}
    gitbook serve {root_rel or "."}
    ```

    3. 构建静态网站：

    ```bash
    gitbook build {root_rel or "."} _book
    ```

    构建结果将输出到 `_book/` 目录，可部署到 GitHub Pages、Netlify、Vercel 或任意静态服务器。

    ## GitBook 云端集成（可选）

    - 在 gitbook.com 创建一个 Space
    - 绑定本仓库与指定分支（如 `main`）
    - 将 GitBook root 配置为 `{root_rel or "."}`，即可自动识别 README 与 SUMMARY

    ## 协议

    本书在**非商业用途**下可免费使用和修改，任何商业使用需要获得作者书面授权。
    详情见 [`LICENSE`](LICENSE)。
    """)


def generate_book_readme(project_name: str) -> str:
    return textwrap.dedent(f"""\
    # {project_name} · 前言

    本书通过 GitBook 构建与发布，你现在看到的是本书的首页（GitBook root 下的 README.md）。

    ## 本书面向谁？

    - 在这里写：你的目标读者，例如：
      - 希望系统化掌握某个领域知识的工程师 / 产品经理 / 学生
      - 对 XX 技术有一定基础，想要进阶的从业者等

    ## 本书希望解决什么问题？

    - 在这里写：本书试图回答的关键问题、弥补的知识断层
    - 尽量具体，而不是泛泛而谈

    ## 如何阅读本书？

    - 从左侧目录（基于 `SUMMARY.md`）选择章节
    - 推荐的阅读顺序 / 不同读者的路径建议
    - 若仓促阅读，可先看哪些章节

    ## 如何参与本书建设？

    - 在 GitHub 仓库提交 Issue 或 Pull Request
    - 参考仓库根目录中的 `CONTRIBUTING.md`
    """)


def generate_contributing() -> str:
    return textwrap.dedent("""\
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
    """)


def generate_code_of_conduct() -> str:
    return textwrap.dedent("""\
    # 行为准则（Code of Conduct）

    为了营造一个友好、开放、包容的社区环境，本项目参与者需遵守以下原则：

    - 尊重他人，不进行人身攻击、歧视或骚扰
    - 接纳不同观点，理性讨论
    - 对他人的付出表示感谢和尊重
    - 遇到问题时，优先尝试友好沟通

    维护者有权移除违反本行为准则的内容或限制相关账户的参与。

    若你在使用或参与本项目的过程中遭遇不当行为，请联系维护者，或在 Issue 中进行反馈（如合适）。
    """)


def generate_license_commercial(project_name: str, author: str = "Your Name") -> str:
    """
    生成「非商业可用，商业使用需授权」的自定义协议。
    可根据需要继续调整条款内容。
    """
    return textwrap.dedent(f"""\
    {project_name} 使用许可协议（需商业授权）

    Copyright (c) 2025 {author}

    一、许可范围

    1.1 在遵守本协议条款的前提下，您可以在**非商业用途**下免费使用、复制、修改、
        展示和分发本项目的全部或部分内容，包括但不限于源文件、文档、示例代码等。

    1.2 非商业用途是指不直接或间接以营利为目的的使用行为，包括但不限于：
        - 个人学习与研究；
        - 在非盈利组织或教育场景中的使用；
        - 在不以盈利为目的的公开分享与演示。

    1.3 任何**商业用途**（包括但不限于下列情形）必须事先获得版权所有者的**书面授权**：
        - 将本项目内容整体或部分用于收费产品或服务；
        - 在公司内部作为正式生产系统、商业解决方案或客户交付的一部分；
        - 以本项目内容为基础进行再创作，并作为商业产品、课程、咨询服务等的一部分进行销售或收费使用；
        - 其他以营利或商业利益为目的的使用方式。

    二、知识产权

    2.1 本项目及其文档中的所有版权、商标权及其他相关知识产权，均由原作者或其指定权利人享有。

    2.2 未经书面许可，您不得移除或更改本项目中的版权声明、作者署名或本许可协议。

    三、责任限制

    3.1 本项目及其文档以“现状”提供，在适用法律允许的最大范围内，不附带任何明示或默示的保证，
        包括但不限于对适销性、特定用途适用性及非侵权性的保证。

    3.2 在任何情况下，版权所有者或贡献者均不对因使用或无法使用本项目所导致的任何直接、间接、
        偶然、特殊或后果性损害（包括但不限于业务中断、信息丢失或利润损失）承担责任，
        即使事先已被告知可能发生此类损害。

    四、终止

    4.1 如您违反本协议的任何条款，本协议授权将自动终止，您也将丧失继续使用本项目的权利。

    4.2 终止后，您必须立即停止使用本项目，并销毁已获取的本项目副本（如有）。

    五、其他

    5.1 本协议未尽事宜，由版权所有者保留最终解释权。

    5.2 若您有商业授权需求或其他特殊用途需求，请联系版权所有者获取书面授权。
    """)


def generate_outline_md() -> str:
    return textwrap.dedent("""\
    # 全书大纲（草案）

    > 提示：先写大纲，再逐章填充；本文件可持续迭代。

    ## Part I 基础篇

    - Chapter 1 开源书的理念与收益
    - Chapter 2 面向谁写：目标读者与痛点
    - Chapter 3 开源协议与知识共享

    ## Part II 实战篇

    - Chapter 4 项目规划与目录结构设计
    - Chapter 5 写作流程与协作规范
    - Chapter 6 评审、质量保证与自动化
    - Chapter 7 构建、发布与版本管理

    ## Part III 进阶篇

    - Chapter 8 社区运营与读者参与
    - Chapter 9 成功案例与经验复盘

    ## TODO

    - [ ] 根据实际主题调整章节结构
    - [ ] 为每一章补充「一句话目标」与小结
    """)


def generate_ch01_intro() -> str:
    return textwrap.dedent("""\
    # 第 1 章：引言

    本章介绍本书的背景、目标、面向读者，以及如何高效使用本书。

    ## 1.1 为什么会有这本书？

    - 这里写：你为什么要写这本开源书？
    - 它希望解决读者的哪些具体问题？

    ## 1.2 适合谁来读？

    - 典型读者画像 1
    - 典型读者画像 2

    ## 1.3 本书将如何组织？

    - 简要介绍 Part / Chapter 结构
    - 如何按照不同水平或角色制定阅读路径

    ## 1.4 如何参与本书建设？

    - 勘误 / 建议：通过 Issue
    - 内容贡献：参考 CONTRIBUTING 文档
    """)


def generate_changelog() -> str:
    return textwrap.dedent("""\
    # 更新日志（Changelog）

    所有值得注意的变更会记录在这里。

    ## [Unreleased]

    - 初始化项目结构（GitBook 版）
    - 编写基础文档：README / CONTRIBUTING / CODE_OF_CONDUCT / LICENSE
    - 创建 GitBook root 下的 README、SUMMARY、docs/outline.md 与 docs/ch01-intro.md
    """)


def generate_summary_md() -> str:
    return textwrap.dedent("""\
    # Summary

    * [前言](README.md)

    ## Part I 基础篇
    * [第 1 章：引言](docs/ch01-intro.md)

    ## Part II 待补充
    * [大纲草案](docs/outline.md)
    """)


def generate_book_json(project_name: str, root_rel: str) -> str:
    # GitBook 经典 book.json 配置
    # root 为相对仓库根目录的路径
    root_val = root_rel or "."
    return textwrap.dedent(f"""\
    {{
      "title": "{project_name}",
      "description": "一本基于 GitBook 的开源书项目",
      "author": "Your Name",
      "language": "zh-hans",
      "root": "{root_val}",
      "structure": {{
        "readme": "README.md",
        "summary": "SUMMARY.md"
      }},
      "plugins": [
        "highlight",
        "search-plus"
      ],
      "pluginsConfig": {{
        "search-plus": {{
          "maxIndexSize": 100000
        }}
      }}
    }}
    """)


def generate_ci_workflow(root_rel: str) -> str:
    """
    生成 GitHub Actions CI 配置：
    - 安装 gitbook-cli
    - 在指定 root 路径下安装插件并构建
    """
    root_val = root_rel or "."
    return textwrap.dedent(f"""\
    name: GitBook CI

    on:
      push:
        branches: [ main, master ]
      pull_request:
        branches: [ main, master ]

    jobs:
      build:
        runs-on: ubuntu-latest

        steps:
          - name: Checkout
            uses: actions/checkout@v4

          - name: Set up Node.js
            uses: actions/setup-node@v4
            with:
              node-version: '18'

          - name: Install gitbook-cli
            run: npm install -g gitbook-cli

          - name: Install GitBook plugins
            run: gitbook install {root_val}

          - name: Build GitBook
            run: gitbook build {root_val} _book
    """)


# ========== 主逻辑 ==========

def main():
    parser = argparse.ArgumentParser(
        description="初始化一个基于 GitBook 的开源书项目目录结构。"
    )
    parser.add_argument(
        "project_name",
        nargs="?",
        help="项目名称（将作为目录名，例如 my-open-book）"
    )
    parser.add_argument(
        "-r", "--root",
        dest="root",
        default=".",
        help="GitBook root 路径，相对于项目根，例如 '.', 'book', 'docs/book'（默认 '.'）"
    )

    args = parser.parse_args()

    # 获取项目名（支持交互输入）
    if args.project_name:
        project_name = args.project_name.strip()
    else:
        try:
            project_name = input("请输入项目名称（将作为目录名，例如 my-open-book）: ").strip()
        except EOFError:
            project_name = ""

    if not project_name:
        print("[错误] 未指定项目名称，已退出。")
        sys.exit(1)

    # 规范化 root 路径
    root_rel = (args.root or ".").strip().strip("\\/")  # 去掉首尾斜杠
    if root_rel == "":
        root_rel = "."

    project_root = Path(project_name).resolve()

    if project_root.exists() and any(project_root.iterdir()):
        print(f"[错误] 目标目录已存在且非空: {project_root}")
        print("为避免覆盖，请更换 project_name 或手动清理目录后再运行。")
        sys.exit(1)

    # 计算 GitBook root 目录
    if root_rel == ".":
        book_root = project_root
    else:
        book_root = project_root / root_rel

    print(f"[信息] 项目根目录: {project_root}")
    print(f"[信息] GitBook root 路径: {root_rel} -> {book_root}")

    # 创建基础目录
    (project_root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    book_root.mkdir(parents=True, exist_ok=True)
    (book_root / "docs").mkdir(parents=True, exist_ok=True)
    (book_root / "assets").mkdir(parents=True, exist_ok=True)

    # 仓库层文件
    create_file(project_root / "README.md", generate_repo_readme(project_name, root_rel))
    create_file(project_root / "CONTRIBUTING.md", generate_contributing())
    create_file(project_root / "CODE_OF_CONDUCT.md", generate_code_of_conduct())
    create_file(
        project_root / "LICENSE",
        generate_license_commercial(project_name, author="Your Name")
    )
    create_file(project_root / "CHANGELOG.md", generate_changelog())
    create_file(project_root / "book.json", generate_book_json(project_name, root_rel))
    create_file(
        project_root / ".github" / "workflows" / "gitbook-ci.yml",
        generate_ci_workflow(root_rel)
    )

    # GitBook root 层文件
    create_file(book_root / "README.md", generate_book_readme(project_name))
    create_file(book_root / "SUMMARY.md", generate_summary_md())
    create_file(book_root / "docs" / "outline.md", generate_outline_md())
    create_file(book_root / "docs" / "ch01-intro.md", generate_ch01_intro())

    print("\n[完成] 基于 GitBook 的开源书项目骨架已创建。")
    print(f"项目路径: {project_root}")
    print(f"GitBook root: {book_root}")
    print("\n下一步建议：")
    print("  1) 编辑 GitBook root 下的 README.md 和 SUMMARY.md，调整前言与目录；")
    print("  2) 在 docs/ 目录中按实际主题扩展章节；")
    print("  3) 在项目根目录运行：")
    print(f"       npm install -g gitbook-cli")
    print(f"       gitbook install {root_rel or '.'}")
    print(f"       gitbook serve {root_rel or '.'}")


if __name__ == "__main__":
    main()
