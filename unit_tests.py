import json
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from parallel_biqbin import ParallelBiqbin
from bab_functions import BabFunctions
from biqbin_data_objects import BiqbinParameters

# Load the JSON with expected results
with open("test/evaluate_results.json", "r") as f:
    data = json.load(f)

ROOT_NODE_BOUNDS = data["root_node_bounds"]
SEED_TEST_DATA = data["seed_test_data"]

# params = BiqbinParameters()
biqbin = ParallelBiqbin()


@pytest.mark.root_bound
@pytest.mark.parametrize("graph_path, expected_bound", ROOT_NODE_BOUNDS.items())
def test_root_node_bound(graph_path, expected_bound):
    adj, num_verts, _, _ = biqbin.read_maxcut_input(graph_path)
    L_matrix = biqbin.get_Laplacian_matrix(adj)

    babfun = BabFunctions(L_matrix, num_verts, biqbin.params)

    biqbin.set_parameters(params=biqbin.params)

    num_test_nodes = 3
    print(f"Graph: {graph_path}: ")
    for i in range(1, num_test_nodes + 1):
        biqbin.set_random_seed(2020)
        globals = biqbin.get_globals(L_matrix, num_verts)

        node_pointer = babfun.generate_node()
        biqbin.evaluate(node_pointer, globals, 0)
        biqbin.free_globals(globals)

        print(
            f"Node {i} Upper bound diff: {round(node_pointer.upper_bound, 2) - expected_bound:.2f}")
        # Allow small floating point tolerance
        assert abs(round(node_pointer.upper_bound, 2) - expected_bound) < 1e-5, \
            f"Upper bound mismatch for {graph_path}"


@pytest.mark.seed_tests
@pytest.mark.parametrize("graph_data", SEED_TEST_DATA)
def test_seed_results(graph_data):
    """
    Test multiple seeds on a given graph and compare the expected bounds.
    """
    graph_path = graph_data["graph"]
    csr_dict = graph_data["csr_matrix"]
    adj = csr_matrix(
        (csr_dict["values"],
         csr_dict["indices"],
         csr_dict["indptr"]),
        shape=tuple(csr_dict["shape"])
    ).toarray()
    num_verts = adj.shape[0]  # Fix: Get number of vertices
    L_matrix = biqbin.get_Laplacian_matrix(adj)

    babfun = BabFunctions(L_matrix, num_verts, biqbin.params)
    biqbin.set_parameters(biqbin.params)

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
