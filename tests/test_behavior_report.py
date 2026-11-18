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
