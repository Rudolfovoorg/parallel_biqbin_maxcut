"""Human-readable comparisons with the recorded solver run (diagnostics only)."""

import json


def behavioral_fields(result):
    fields = {}
    for kind in ("maxcut", "qubo", "bqp"):
        for key, value in result.get(kind, {}).items():
            fields[f"{kind}.{key}"] = value
    metadata = result["meta_data"]
    for key in ("time", "eval_bab_nodes", "num_workers_used"):
        if key in metadata:
            fields[f"meta_data.{key}"] = metadata[key]
    for prefix, data in (("meta_data", metadata), ("root", metadata["root_node"])):
        # Older reference files use heuristic_run_count for the same counter.
        count = data.get("heuristic_call_count", data.get("heuristic_run_count"))
        if count is not None:
            fields[f"{prefix}.heuristic_call_count"] = count
    root = metadata["root_node"]
    for key in ("sdp_value", "heuristic_value", "root_solution"):
        if key in root:
            fields[f"root.{key}"] = root[key]
    if "heuristic_data" in root:
        fields["root.heuristic_data.length"] = len(root["heuristic_data"])
        fields["root.heuristic_data.values"] = [entry["value"] for entry in root["heuristic_data"]]
    for key, value in metadata.get("custom_solver_tests", {}).items():
        fields[f"custom_solver_tests.{key}"] = value
    return fields


def compare_behavior(instance, expected, actual):
    """Return a text report and changed-field count; never assert on differences."""
    expected = behavioral_fields(expected)
    actual = behavioral_fields(actual)
    missing = object()
    changed = []
    unchanged = []
    for key in sorted(expected.keys() | actual.keys()):
        before = expected.get(key, missing)
        after = actual.get(key, missing)
        if before == after:
            unchanged.append(key)
            continue
        lines = [f"CHANGED {key}"]
        for label, value in (("expected", before), ("actual", after)):
            text = "<missing>" if value is missing else json.dumps(value)
            lines.append(f"  {label}: {text}")
        if type(before) in (int, float) and type(after) in (int, float):
            lines.append(f"  delta (actual - expected): {after - before:+.12g}")
        if (
            key.endswith((".x", ".root_solution"))
            and isinstance(before, list)
            and isinstance(after, list)
        ):
            positions = [i + 1 for i, (a, b) in enumerate(zip(before, after)) if a != b]
            positions.extend(
                range(min(len(before), len(after)) + 1, max(len(before), len(after)) + 1)
            )
            lines.append(f"  changed vector positions (1-based): {positions}")
        changed.append("\n".join(lines))
    header = [
        f"Instance: {instance}",
        f"Behavior: {len(changed)} changed fields; {len(unchanged)} unchanged fields",
        "Differences are informational. Correctness assertions run separately.",
    ]
    body = "\n\n".join(changed) if changed else "No behavioral differences."
    return "\n".join(header) + "\n\n" + body + "\n\nUNCHANGED: " + ", ".join(unchanged) + "\n", len(
        changed
    )
