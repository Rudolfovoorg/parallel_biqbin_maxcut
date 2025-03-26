import json
import pytest
from parallel_biqbin import ParallelBiqbin
from helper_functions import HelperFunctions
from parallel_biqbin_maxcut import BabFunctions
from biqbin_data_objects import GlobalVariables

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

    biqbin.biqbin.set_BabPbSize(num_verts)
    biqbin.set_parameters(params)

    num_test_nodes = 3
    for i in range(1, num_test_nodes + 1):
        biqbin.set_random_seed(2020)
        globals = biqbin.get_globals(L_matrix, num_verts)

        node = babfun.generate_node()
        node.upper_bound = biqbin.evaluate(node, globals, 0)
        biqbin.free_globals(globals)

        print(f"Graph: {graph_path} node {i}:")
        print(
            f"Upper bound diff: {round(node.upper_bound, 2) - expected_bound:.2f};")
        # Allow small floating point tolerance
        assert abs(round(node.upper_bound, 2) == expected_bound), \
            f"Upper bound mismatch for {graph_path}"
