import json
import pytest
from parallel_biqbin import ParallelBiqbin
from helper_functions import HelperFunctions
from parallel_biqbin_maxcut import BabFunctions

# Load the json with expected root node bounds
with open("test/evaluate_results.json", "r") as f:
    EXPECTED_BOUNDS = json.load(f)["root_node_bounds"]


@pytest.mark.parametrize("graph_path,expected_bound", EXPECTED_BOUNDS.items())
def test_root_node_bound(graph_path, expected_bound):
    biqbin = ParallelBiqbin()
    helper = HelperFunctions()

    params_path = "test/params"
    params = helper.read_parameters_file(params_path)

    adj, num_verts, _, _ = helper.read_maxcut_input(graph_path)
    L_matrix = helper.get_SP_L_matrix(adj)

    babfun = BabFunctions(L_matrix, num_verts, params)

    biqbin.unit_test_init(L_matrix, num_verts, params)
    node = babfun.generate_node()
    biqbin.evaluate_node(node, 0)

    print(f"\nGraph: {graph_path} root node: Computed upper bound: {node.upper_bound:.2f} - Expected upper bound: {expected_bound:.2f}")

    # Allow small floating point tolerance
    assert abs(node.upper_bound - expected_bound) < 1e-2, \
        f"Upper bound mismatch for {graph_path}"
