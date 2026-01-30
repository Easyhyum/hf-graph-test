import argparse
import os
import re
from typing import Iterable, Tuple


NODE_BLOCK_RE = re.compile(r"\"([^\"]+)\"\s*\[.*?label=\"(.*?)\"\s*\];", re.DOTALL)
ID_RE = re.compile(r"\bID\s*\|\s*(\d+)")
TRIPLE_RE = re.compile(r"(?:<<<|\\<\\<\\<)\s*([0-9,\s\{\}]+)\s*(?:>>>|\\>\\>\\>)")
NODE_NAME_ID_RE = re.compile(r"(?:^|_)node_(\d+)")


def _parse_launch_items(launch: str):
    items = []
    buf = ""
    in_brace = False
    brace_buf = ""

    for ch in launch:
        if ch == "{" and not in_brace:
            in_brace = True
            brace_buf = ""
            continue
        if ch == "}" and in_brace:
            in_brace = False
            vals = [int(x.strip()) for x in brace_buf.split(",") if x.strip()]
            items.append(vals)
            brace_buf = ""
            continue
        if ch == "," and not in_brace:
            if buf.strip():
                items.append(int(buf.strip()))
            buf = ""
            continue

        if in_brace:
            brace_buf += ch
        else:
            buf += ch

    if buf.strip():
        items.append(int(buf.strip()))

    return items


def _parse_launch_config(label: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
    normalized = label.replace("\\<", "<").replace("\\>", ">").replace("\\{", "{").replace("\\}", "}")
    match = TRIPLE_RE.search(normalized)
    if not match:
        return (0, 0, 0), (0, 0, 0), 0

    launch = match.group(1)
    items = _parse_launch_items(launch)

    grid_item = items[0] if len(items) > 0 else None
    block_item = items[1] if len(items) > 1 else None
    shared_item = items[2] if len(items) > 2 else None

    if isinstance(grid_item, list):
        grid = (
            grid_item[0] if len(grid_item) > 0 else 1,
            grid_item[1] if len(grid_item) > 1 else 1,
            grid_item[2] if len(grid_item) > 2 else 1,
        )
    elif isinstance(grid_item, int):
        grid = (grid_item, 1, 1)
    else:
        grid = (0, 0, 0)

    if isinstance(block_item, list):
        block = (
            block_item[0] if len(block_item) > 0 else 1,
            block_item[1] if len(block_item) > 1 else 1,
            block_item[2] if len(block_item) > 2 else 1,
        )
    elif isinstance(block_item, int):
        block = (block_item, 1, 1)
    else:
        block = (0, 0, 0)

    shared = shared_item if isinstance(shared_item, int) else 0

    return grid, block, shared


def _parse_node_id(node_name: str, label: str) -> str:
    match = ID_RE.search(label)
    if match:
        return match.group(1)

    match = NODE_NAME_ID_RE.search(node_name)
    if match:
        return match.group(1)

    return node_name


def parse_dot_file(dot_path: str) -> Iterable[str]:
    with open(dot_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for node_name, label in NODE_BLOCK_RE.findall(content):
        node_id = _parse_node_id(node_name, label)
        grid, block, shared = _parse_launch_config(label)
        yield f"Node {node_id}: Grid ({grid[0]},{grid[1]},{grid[2]}), Block ({block[0]},{block[1]},{block[2]}) shared: {shared}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CUDA graph DOT files into text summaries.")
    parser.add_argument(
        "--dir",
        default=None,
        help="Directory containing .dot files. Defaults to ./cuda_graphs next to this script if present, otherwise cwd.",
    )
    args = parser.parse_args()

    base_dir = args.dir
    if base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "cuda_graphs")
        base_dir = candidate if os.path.isdir(candidate) else os.getcwd()

    for name in os.listdir(base_dir):
        if not name.endswith(".dot"):
            continue
        dot_path = os.path.join(base_dir, name)
        output_name = f"parse_{name}.txt"
        output_path = os.path.join(base_dir, output_name)
        lines = list(parse_dot_file(dot_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
