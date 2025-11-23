#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能说明：
1. 扫描指定根目录及其子目录中，所有指定后缀的文件；
2. 根目录下的目标文件 => 汇总到 <根目录名>.txt（只包含根目录，不包含子目录）；
3. 每个一级子目录及其子目录中的目标文件 => 汇总到 <一级子目录名>.txt（输出在根目录）；
4. 可通过 --extra-dir 指定“特别目录”，会将该目录及其所有子目录中的目标文件，
   汇总到 <该目录名>.txt（同样输出在根目录）。

所有生成的 txt 均放在根目录中。
"""

import os
import argparse


def normalize_exts(raw_exts):
    """
    规范化后缀列表：
    - 支持 --ext .md .markdown
    - 也支持 --ext md markdown
    - 也支持 --ext .md,.markdown
    转成：['.md', '.markdown'] 且小写、去重
    """
    if not raw_exts:
        return ['.md', '.markdown']

    exts = []
    for item in raw_exts:
        # 支持逗号分隔
        parts = item.split(',')
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not p.startswith('.'):
                p = '.' + p
            p = p.lower()
            if p not in exts:
                exts.append(p)
    return exts


def is_target_file(filename: str, target_exts):
    """根据后缀判断文件是否是目标文件"""
    _, ext = os.path.splitext(filename)
    return ext.lower() in target_exts


def merge_texts(root_dir: str, target_exts, extra_dirs):
    root_dir = os.path.abspath(root_dir)
    root_basename = os.path.basename(root_dir)

    # 1. 根目录下的目标文件内容
    root_contents = []

    # 2. 一级子目录：<目录名> -> 内容块列表
    first_level_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    first_level_map = {d: [] for d in first_level_dirs}

    # 3. 特别目录：准备规格列表 & 内容 map（以“输出文件名”为 key）
    #    extra_dirs 可以是绝对路径，也可以是相对 root_dir 的路径
    special_specs = []  # 列表元素为 (rel_path, output_name)
    if extra_dirs:
        for raw_path in extra_dirs:
            # 处理成绝对路径
            if os.path.isabs(raw_path):
                abs_path = os.path.abspath(raw_path)
            else:
                abs_path = os.path.abspath(os.path.join(root_dir, raw_path))

            if not os.path.isdir(abs_path):
                print(f"[WARN] 特别目录不存在或不是目录，已跳过: {raw_path}")
                continue

            # 确保在 root_dir 内
            if not (abs_path == root_dir or abs_path.startswith(root_dir + os.sep)):
                print(f"[WARN] 特别目录不在根目录内，已跳过: {raw_path}")
                continue

            rel_path = os.path.relpath(abs_path, root_dir)  # 例如 "a/a1" 或 "."
            output_name = os.path.basename(rel_path.rstrip(os.sep)) if rel_path != "." else root_basename

            # 避免跟根目录汇总同名，防止覆盖
            if output_name == root_basename and rel_path != ".":
                print(f"[WARN] 特别目录 '{raw_path}' 的输出名与根目录同名，已跳过以避免覆盖: {output_name}.txt")
                continue

            special_specs.append((rel_path, output_name))

    # 为每个特别目录名准备一个列表
    special_map = {}
    for rel_path, out_name in special_specs:
        if out_name not in special_map:
            special_map[out_name] = []

    # 4. 遍历目录树，收集内容
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)  # "." 或 "a" 或 "a/a1" 之类

        for fname in filenames:
            if not is_target_file(fname, target_exts):
                continue

            full_path = os.path.join(dirpath, fname)

            # 读文件内容
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            # 用相对根目录的路径作为标题标签
            rel_file_for_show = os.path.relpath(full_path, root_dir)
            block = f"# === {rel_file_for_show} ===\n\n{content}\n\n"

            # 4.1 根目录汇总：仅限根目录下的文件（rel_dir == "."）
            if rel_dir == ".":
                root_contents.append(block)
            else:
                # 4.2 一级子目录汇总：根据 top-level 目录名分组
                parts = rel_dir.split(os.sep)
                top_folder = parts[0]
                if top_folder in first_level_map:
                    first_level_map[top_folder].append(block)

            # 4.3 特别目录汇总：看文件是否位于某个指定目录下
            rel_file_path = os.path.relpath(full_path, root_dir)  # 比如 "a/a1/a1.md"
            for spec_rel, out_name in special_specs:
                # spec_rel 可能为 "."、"a"、"a/a1"
                if spec_rel == ".":
                    # 特别目录是根目录：则包含整个树
                    special_map[out_name].append(block)
                else:
                    # 判断 rel_file_path 是否在 spec_rel 之下
                    if (
                        rel_file_path == spec_rel
                        or rel_file_path.startswith(spec_rel + os.sep)
                    ):
                        special_map[out_name].append(block)

    # 5. 写出根目录汇总 <根目录名>.txt（只包含根目录的文件）
    if root_contents:
        root_txt_path = os.path.join(root_dir, f"{root_basename}.txt")
        with open(root_txt_path, "w", encoding="utf-8") as f:
            f.writelines(root_contents)
        print(f"[OK] 根目录目标文件已汇总到: {root_txt_path}")
    else:
        print("[INFO] 根目录下没有找到任何目标文件（指定后缀），不生成根目录汇总 txt。")

    # 6. 写出一级子目录汇总 <一级目录名>.txt（放在根目录）
    for folder_name, blocks in first_level_map.items():
        if not blocks:
            print(f"[INFO] 一级子目录 '{folder_name}' 下没有目标文件，跳过生成 txt。")
            continue

        out_path = os.path.join(root_dir, f"{folder_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(blocks)
        print(f"[OK] 目录 '{folder_name}' 及其子目录目标文件已汇总到: {out_path}")

    # 7. 写出特别目录汇总 <目录名>.txt（放在根目录）
    for out_name, blocks in special_map.items():
        if not blocks:
            print(f"[INFO] 特别目录汇总 '{out_name}' 下没有目标文件，跳过生成 txt。")
            continue

        # 避免和根目录汇总重名（这里只再防一次双保险）
        if out_name == root_basename:
            out_file = f"{out_name}_extra.txt"
        else:
            out_file = f"{out_name}.txt"

        out_path = os.path.join(root_dir, out_file)
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(blocks)
        print(f"[OK] 特别目录 '{out_name}' 目标文件已汇总到: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "扫描指定根目录及其子目录中的文件，"
            "根据后缀汇总：\n"
            "1) 根目录下的文件 => <根目录名>.txt\n"
            "2) 每个一级子目录及其子目录 => <一级子目录名>.txt\n"
            "3) --extra-dir 指定的目录及其子目录 => <目录名>.txt\n"
            "所有 txt 输出在根目录。"
        )
    )
    parser.add_argument(
        "root_dir",
        help="要扫描的根目录路径"
    )
    parser.add_argument(
        "-e", "--ext",
        nargs="+",
        help="指定要提取的文件后缀，例如：-e .md .markdown 或 -e md markdown 或 -e .md,.markdown；默认只处理 .md .markdown",
    )
    parser.add_argument(
        "-d", "--extra-dir",
        action="append",
        help="特别目录，可以多次使用，例如：-d docs -d notes/ch1；会将该目录及其子目录内容汇总到 <目录名>.txt 中（输出在根目录）",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"[ERROR] 指定路径不是有效目录: {args.root_dir}")
        return

    exts = normalize_exts(args.ext)
    print(f"[INFO] 使用的文件后缀过滤: {exts}")

    merge_texts(args.root_dir, exts, args.extra_dir)


if __name__ == "__main__":
    main()
