"""One-off verification: the overridden `_generate_and_score_completions` in
rl_grpo.py must be a copy of the installed trl original except at the three
marked SITE locations (1a/1b typed expansion + n=1 generation, 2 advantage).

Method: parse both files with `ast`, compare the two method bodies node by
node. Nodes whose `ast.dump` differs are reported; differences are only
accepted inside the SITE blocks (detected by `mm_gen_mode` / `advantage_mode`
in the dump of an `If` node).
"""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import glob

TRL_GRPO = Path(sys.prefix) / "lib/python3.12/site-packages/trl/trainer/grpo_trainer.py"
if not TRL_GRPO.exists():
    cands = glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/trl/trainer/grpo_trainer.py"))
    TRL_GRPO = Path(cands[0]) if cands else TRL_GRPO

METHOD = "_generate_and_score_completions"


def get_method(path: Path) -> ast.FunctionDef | None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == METHOD:
            return node
    return None


def collect_diffs(node_a: ast.AST, node_b: ast.AST, path: str, out: list[str]) -> None:
    if ast.dump(node_a) == ast.dump(node_b):
        return
    # SITE blocks: If nodes whose test mentions the new kwargs — intentional.
    if isinstance(node_a, ast.If) and isinstance(node_b, ast.If):
        da, db = ast.dump(node_a.test), ast.dump(node_b.test)
        if "mm_gen_mode" in da or "advantage_mode" in da:
            out.append(f"SITE-OK {path}")
            return
    da, db = ast.dump(node_a), ast.dump(node_b)
    out.append(f"DIFF  {path}\n  A: {da[:200]}\n  B: {db[:200]}")
    # recurse into children for a finer-grained report
    for child_a, child_b in zip(ast.iter_child_nodes(node_a), ast.iter_child_nodes(node_b)):
        collect_diffs(child_a, child_b, f"{path}.{type(child_a).__name__}", out)


def main() -> int:
    ours = get_method(REPO / "src/agents_memory/rl_grpo.py")
    theirs = get_method(TRL_GRPO)
    if ours is None or theirs is None:
        print(f"method not found: ours={ours is not None} theirs={theirs is not None}")
        return 1

    # Compare body statements pairwise.
    reports: list[str] = []
    body_a, body_b = ours.body, theirs.body
    n = max(len(body_a), len(body_b))
    for i in range(n):
        stmt_a = body_a[i] if i < len(body_a) else None
        stmt_b = body_b[i] if i < len(body_b) else None
        if stmt_a is None or stmt_b is None:
            reports.append(f"DIFF  body[{i}] missing on one side: A={stmt_a is not None} B={stmt_b is not None}")
            continue
        collect_diffs(stmt_a, stmt_b, f"body[{i}]", reports)

    sites = [r for r in reports if r.startswith("SITE-OK")]
    real = [r for r in reports if r.startswith("DIFF")]
    print(f"trl source: {TRL_GRPO}")
    print(f"body statements: ours={len(body_a)} theirs={len(body_b)}")
    print(f"SITE blocks accepted: {len(sites)}")
    print(f"unexpected diffs: {len(real)}")
    for r in real:
        print(r)
    return 0 if not real else 1


if __name__ == "__main__":
    sys.exit(main())
