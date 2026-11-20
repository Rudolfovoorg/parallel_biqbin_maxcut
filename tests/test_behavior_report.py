from copy import deepcopy

from tests.behavior_report import compare_behavior


def reference():
    return {
        "maxcut": {"computed_val": 2, "x": [1, 0, 0], "solution": [1]},
        "meta_data": {
            "eval_bab_nodes": 1,
            "heuristic_run_count": 4,
            "root_node": {
                "sdp_value": 2.5,
                "heuristic_value": 2,
                "heuristic_run_count": 4,
                "root_solution": [1, 0, 0],
                "heuristic_data": [{"value": 2}],
            },
        },
    }


def test_counter_aliases_do_not_produce_false_differences():
    expected = reference()
    actual = deepcopy(expected)
    for data in (actual["meta_data"], actual["meta_data"]["root_node"]):
        data["heuristic_call_count"] = data.pop("heuristic_run_count")
    text, count = compare_behavior("example", expected, actual)
    assert count == 0
    assert "No behavioral differences" in text


def test_reports_values_deltas_and_changed_vector_positions():
    expected = reference()
    actual = deepcopy(expected)
    actual["maxcut"]["x"] = [0, 1, 0]
    actual["meta_data"]["root_node"]["sdp_value"] = 3.0
    text, count = compare_behavior("example", expected, actual)
    assert count == 2
    assert "CHANGED maxcut.x" in text
    assert "expected: [1, 0, 0]" in text
    assert "actual: [0, 1, 0]" in text
    assert "changed vector positions (1-based): [1, 2]" in text
    assert "delta (actual - expected): +0.5" in text
    assert expected == reference()


def test_reports_missing_fields_and_vector_length_changes():
    expected = reference()
    actual = deepcopy(expected)
    del actual["meta_data"]["root_node"]["heuristic_data"]
    actual["maxcut"]["x"] = [1, 0]
    text, count = compare_behavior("example", expected, actual)
    assert count == 3
    assert "actual: <missing>" in text
    assert "changed vector positions (1-based): [3]" in text


def test_reports_survive_failure_reset_on_rerun_and_stay_separate(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "maxcut": {
                    "shape": [3, 3],
                    "nnz": 6,
                    "row": [0, 0, 1, 1, 2, 2],
                    "col": [1, 2, 0, 2, 0, 1],
                    "data": [1] * 6,
                }
            }
        )
    )
    expected = reference()
    expected["meta_data"].update(instance=str(graph), time=0.1, parameters={"gcd": 1})
    expected["meta_data"]["root_node"]["heuristic_data"] = [{"time": 0.01, "value": 2}]
    expected["meta_data"]["root_node"]["heuristic_run_count"] = 1
    actual = deepcopy(expected)
    for section in (actual["meta_data"], actual["meta_data"]["root_node"]):
        section["heuristic_call_count"] = section.pop("heuristic_run_count")
    instances = [tmp_path / "first", tmp_path / "second"]
    for instance in instances:
        Path(str(instance) + "-expected_output.json").write_text(json.dumps(expected))
        Path(str(instance) + ".output.json").write_text(json.dumps(actual))
    broken = deepcopy(actual)
    broken["maxcut"]["computed_val"] = 3
    Path(str(instances[0]) + ".output.json").write_text(json.dumps(broken))
    reports = [tmp_path / "target-a.txt", tmp_path / "target-b.txt"]
    reports[0].write_text("stale report")
    env = os.environ.copy()
    # Never reuse the parent's --basetemp or coverage append settings.
    env.pop("PYTEST_ADDOPTS", None)

    def run(report):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_biqbin_output.py",
                "-q",
                "--behavior-report",
                str(report),
                "--instances",
                *map(str, instances),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    failed = run(reports[0])
    assert failed.returncode == 1, failed.stdout + failed.stderr
    failed_report = reports[0].read_text()
    assert "CHANGED maxcut.computed_val" in failed_report
    assert "stale report" not in failed_report
    assert failed_report.count("Instance:") == 2
    Path(str(instances[0]) + ".output.json").write_text(json.dumps(actual))
    passed = run(reports[1])
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert reports[0].read_text() == failed_report
    rerun = run(reports[0])
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert reports[0].read_text() == reports[1].read_text()
    assert reports[0].read_text().count("Instance:") == 2
    assert "CHANGED maxcut.computed_val" not in reports[0].read_text()
