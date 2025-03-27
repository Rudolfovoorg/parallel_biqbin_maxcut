import json
import pytest
import numpy as np
from parallel_biqbin import ParallelBiqbin
from helper_functions import HelperFunctions
from parallel_biqbin_maxcut import BabFunctions

# Load the JSON with expected results
with open("test/evaluate_results.json", "r") as f:
    data = json.load(f)

ROOT_NODE_BOUNDS = data["root_node_bounds"]
SEED_TEST_DATA = data["seed_test_data"]


@pytest.mark.parametrize("graph_path, expected_bound", ROOT_NODE_BOUNDS.items())
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
    print(f"Graph: {graph_path}: ")
    for i in range(1, num_test_nodes + 1):
        biqbin.set_random_seed(2020)
        globals = biqbin.get_globals(L_matrix, num_verts)

        node = babfun.generate_node()
        node.upper_bound = biqbin.evaluate(node, globals, 0)
        biqbin.free_globals(globals)

        print(
            f"Node {i} Upper bound diff: {round(node.upper_bound, 2) - expected_bound:.2f}")
        # Allow small floating point tolerance
        assert abs(round(node.upper_bound, 2) - expected_bound) < 1e-5, \
            f"Upper bound mismatch for {graph_path}"


@pytest.mark.parametrize("graph_data", SEED_TEST_DATA)
def test_seed_results(graph_data):
    """
    Test multiple seeds on a given graph and compare the expected bounds.
    """
    biqbin = ParallelBiqbin()
    helper = HelperFunctions()

    params_path = "test/params"
    params = helper.read_parameters_file(params_path)

    graph_path = graph_data["graph"]
    adj = np.array(graph_data["adjacency_matrix"])
    num_verts = adj.shape[0]  # Fix: Get number of vertices
    L_matrix = helper.get_SP_L_matrix(adj)

    babfun = BabFunctions(L_matrix, num_verts, params)

    biqbin.biqbin.set_BabPbSize(num_verts)
    biqbin.set_parameters(params)

    tests = graph_data["tests"]
    print(f"Graph {graph_path}")
    for test in tests:
        biqbin.set_random_seed(test["seed"])
        globals = biqbin.get_globals(L_matrix, num_verts)

        node = babfun.generate_node()
        result = biqbin.evaluate(node, globals, 0)
        biqbin.free_globals(globals)
        solX = np.frombuffer(
            node.sol.X,
            dtype=np.int32,
            count=num_verts
        ).astype(np.int32)

        solution_nodes = []
        for i in range(num_verts):
            if solX[i] == 1:
                solution_nodes.append(i + 1)

        expected_bound = test["result"]
        expected_solution_nodes = test["solution_nodes"]
        print(
            f"Seed {test['seed']}: Upper bound diff: {result - expected_bound}")
        print(f"Solution ({len(solution_nodes)}) = {solution_nodes}")

        # Allow small floating point tolerance
        assert abs(
            result - expected_bound) < 1e-5, f"Upper bound mismatch for {graph_path}, Seed {test['seed']}"
        # Check solution vector'
        assert len(solution_nodes) == len(
            expected_solution_nodes), f"Number of nodes in solution is different for {graph_path}, seed {test['seed']}"
        assert all([solution_nodes[i] == expected_solution_nodes[i] for i in range(len(
            solution_nodes))]), f"Wrong solution nodes for {graph_path}, seed {test['seed']}"
