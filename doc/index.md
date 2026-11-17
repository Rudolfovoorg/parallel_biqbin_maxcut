
# BiqBin

BiqBin is a parallel branch-and-bound solver for Max-Cut
and binary quadratic optimization problems.

It provides:

- A native C solver using MPI
- Python bindings through pybind11
- Support for Max-Cut, QUBO and general BQP problems
- Configurable logging and per-rank diagnostics

## Getting started

See the project's
[README](https://github.com/Rudolfovoorg/parallel_biqbin_maxcut)
for installation, dependencies and usage examples.

## Documentation

Browse the documentation navigation for supported
problem formats, configuration and examples.

## Python API

The [API reference](api/index.md) is generated from the Python source and
native binding type stubs. It includes the solver classes, file parsers,
logging helpers, utilities, and callback interfaces.

## Building these docs

From the repository root:

```bash
python -m pip install -r requirements-docs.txt
make docs
make docs-serve
```

`make docs` builds into `site/` and treats warnings as errors. `make docs-serve`
starts a local preview and watches the Python sources as well as these guides.
Neither command publishes or deploys anything. A compiled solver, NumPy,
SciPy, and MPI are not required to build the documentation.

`doc/` contains the guide and reference-page Markdown. The `mkdocstrings`
plugin reads Python code from the repository root; `docs_dir` does not limit
which Python packages can be documented. Edit docstrings in the source files
and rebuild to update the API reference. Add new modules to the API pages and
navigation in `mkdocs.yml`.
