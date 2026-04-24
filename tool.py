"""
集中存放项目里会用到的 LangChain Tool（`@tool` 装饰的函数）。
文件类工具仅允许操作与本文件同级的 `file` 目录下的路径。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool


def _file_root() -> Path:
    """
    返回受控根目录：`<项目根>/file`。

    @returns {Path} 绝对路径。
    """
    return Path(__file__).resolve().parent / "file"


def _basename_only(name: str) -> str:
    """
    仅保留一层文件名，禁止子路径与 `..`。

    @param {string} name 用户传入的名称。
    @returns {string} 纯文件名。
    @throws {ValueError} 若包含路径分隔或父目录引用。
    """
    if not name or name.strip() != name:
        raise ValueError("文件名不能为空或含首尾空白。")
    p = Path(name)
    if p.name != name or ".." in p.parts:
        raise ValueError("仅允许单层文件名，不能包含路径或 `..`。")
    return p.name


def _path_in_file_root(name: str) -> Path:
    """
    解析为 `file` 目录下的绝对路径并校验不越界。

    @param {string} name 单层文件名。
    @returns {Path} 绝对路径。
    @throws {ValueError} 若解析后不在 `file` 根下。
    """
    root = _file_root().resolve()
    target = (root / _basename_only(name)).resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ValueError("路径必须位于 `file` 文件夹内。") from e
    return target


@tool
def list_files_in_file_folder() -> str:
    """
    列出 `file` 文件夹下**当前这一层**的所有文件（不含子目录里的文件）。

    @returns {string} 每行一个文件名；若无文件或目录不存在则返回说明文字。
    """
    root = _file_root()
    if not root.is_dir():
        return f"目录不存在或不是文件夹: {root}"
    names = sorted(
        (p.name for p in root.iterdir() if p.is_file()),
        key=lambda n: n.lower(),
    )
    return "\n".join(names) if names else "(当前目录下没有文件)"


@tool
def read_file_in_file_folder(filename: str) -> str:
    """
    以 UTF-8 读取 `file` 文件夹内指定文件的全部文本。

    @param {string} filename 单层文件名，例如 `notes.txt`。
    @returns {string} 文件内容或错误说明。
    """
    path = _path_in_file_root(filename)
    if not path.is_file():
        return f"不是已存在的文件: {path}"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        return f"读取失败: {e}"


@tool
def rename_file_in_file_folder(old_filename: str, new_filename: str) -> str:
    """
    在 `file` 文件夹内重命名文件（不改变目录，仅改文件名）。

    @param {string} old_filename 当前文件名（单层）。
    @param {string} new_filename 新文件名（单层）。
    @returns {string} 成功或失败说明。
    """
    old_path = _path_in_file_root(old_filename)
    new_path = _path_in_file_root(new_filename)
    if not old_path.is_file():
        return f"源文件不存在: {old_path}"
    if new_path.exists():
        return f"目标名已存在: {new_path}"
    try:
        old_path.rename(new_path)
    except OSError as e:
        return f"重命名失败: {e}"
    return f"已将 `{old_path.name}` 重命名为 `{new_path.name}`。"


ALL_TOOLS = [
    list_files_in_file_folder,
    read_file_in_file_folder,
    rename_file_in_file_folder,
]
