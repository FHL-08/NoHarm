"""CTRL-2.3@1 The executor source contains no code path that snaps an
object between the palms.

Contract: inspect Executor.grasp() and every helper it can call. Reject
any reachable write to live MuJoCo pose or velocity buffers. A pose write
inside this path moves the prop outside physics, including a renamed snap
helper.
"""

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "harmless" / "executor.py"
LIVE_STATE = {"qpos", "qvel", "xpos", "xquat"}


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _called_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _grasp_reachable_functions(tree):
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "grasp" in functions, "Executor source has no grasp() entry point"
    reachable = set()
    pending = ["grasp"]
    while pending:
        name = pending.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call):
                called = _called_name(node.func)
                if called in functions and called not in reachable:
                    pending.append(called)
    return {name: functions[name] for name in reachable}


def _live_state_writes(functions):
    aliases = set()
    for fn in functions.values():
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                if value is not None and _chain(value)[-1:] == ("data",):
                    for target in targets:
                        aliases.update(_target_names(target))

    writes = []
    for name, fn in functions.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                base = target.value if isinstance(target, ast.Subscript) else target
                chain = _chain(base)
                if len(chain) < 2 or chain[-1] not in LIVE_STATE:
                    continue
                owner = chain[-2]
                if owner == "data" or owner in aliases:
                    writes.append(f"{name}:{node.lineno} {'.'.join(chain)}")
    return writes


def test_ctrl_2_3_executor_grasp_path_never_writes_live_pose():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    reachable = _grasp_reachable_functions(tree)
    writes = _live_state_writes(reachable)

    assert not writes, (
        "Executor.grasp() can reposition an object outside physics by "
        f"writing live pose or velocity: {writes}"
    )
