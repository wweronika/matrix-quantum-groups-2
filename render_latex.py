from __future__ import annotations

from pathlib import Path
from sympy import preview


def sympy_to_png(expr, output_path: str | Path, *, euler: bool = False, fontsize: int = 14):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview(
        expr,
        output="png",
        viewer="file",
        filename=str(output_path),
        euler=euler,
        fontsize=fontsize,
    )

    return output_path