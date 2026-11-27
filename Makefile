# BiqBin build and development targets
.DEFAULT_GOAL := all

PYTHON ?= python3
RUFF ?= $(PYTHON) -m ruff
MPI_PYTHON ?= $(PYTHON)

# Docker image configuration
IMAGE ?= parallel-biqbin
IMAGE_DEV ?= parallel-biqbin-dev
TAG ?= 1.0.0
DOCKER_BUILD_PARAMS ?=
DATA_DIR ?= tests/

# Build Directories
WRAPPER_BUILD_DIR := build/wrapper
C_BUILD_DIR := build/c_build

# Tests Directories
ARTIFACT_DIR ?= artifacts
COVERAGE_DIR := $(ARTIFACT_DIR)/coverage
TEST_RESULTS_DIR := $(ARTIFACT_DIR)/tests
JUNIT_DIR ?= $(TEST_RESULTS_DIR)/junit

# Compiler
CC = mpicc
CXX = mpic++

LINALG = -lopenblas -lm
OPTI = -O3 -ffast-math -fexceptions -fPIC -fno-common
CPPOPTI = $(OPTI)

# Minimum native log severity (Python logging levels); rebuild after changing it.
LOG_LEVEL ?= 20
LOG_LEVEL_DEBUG = 10
LOG_LEVEL_INFO = 20
LOG_LEVEL_WARNING = 30
LOG_LEVEL_WARN = 30
LOG_LEVEL_ERROR = 40
LOG_LEVEL_CRITICAL = 50
LOG_LEVEL_FATAL = 50

# Accept case-insensitive names or decimal thresholds; leading zeros are rejected.
LOG_LEVEL_NAME := $(shell printf '%s' '$(subst ','"'"',$(strip $(LOG_LEVEL)))' | tr '[:lower:]' '[:upper:]')
LOG_LEVEL_VALUE := $(or $(LOG_LEVEL_$(LOG_LEVEL_NAME)),$(shell printf '%s\n' '$(subst ','"'"',$(LOG_LEVEL_NAME))' | sed -n -e '/^0$$/p' -e '/^[1-9][0-9]*$$/p'))
ifeq ($(LOG_LEVEL_VALUE),)
$(error Invalid LOG_LEVEL '$(LOG_LEVEL)'; use DEBUG, INFO, WARNING, ERROR, CRITICAL, WARN, FATAL, or a nonnegative decimal integer without leading zeros)
endif
LOG_DEFINES = -DBIQBIN_LOG_MIN_LEVEL=$(LOG_LEVEL_VALUE)

# Query Python build configuration only when a native build needs it.
PYTHON_VERSION = $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE = $(shell $(PYTHON)-config --includes)
PYBIND11_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_LDFLAGS = $(shell $(PYTHON)-config --ldflags)
PYTHON_LIBS = $(shell $(PYTHON)-config --libs)
PYTHON_LIB = -lpython$(PYTHON_VERSION) $(PYTHON_LDFLAGS) $(PYTHON_LIBS)
INCLUDES = $(PYBIND11_INCLUDES) $(PYTHON_INCLUDE)
LIB = $(PYTHON_LIB)

# Python module (Pybind11)
PYMODULE := biqbin_module.so
PYMOD_OUT := $(WRAPPER_BUILD_DIR)/$(PYMODULE)
# C only binary
BIQBIN_BINARY := biqbin_executable
BINS := $(C_BUILD_DIR)/$(BIQBIN_BINARY)

# Bound MPI runs, including a forced shutdown if graceful termination hangs.
MPI_TIMEOUT ?= 180
MPI_RUN = timeout --kill-after=10s $(MPI_TIMEOUT)s mpiexec -n 3

# Test targets share output files. Serialize even when invoked with make -j.
ifneq ($(filter test%,$(MAKECMDGOALS)),)
.NOTPARALLEL:
endif

RUN_ENVS = OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1

# Pure native executable objects
C_OBJS = $(C_BUILD_DIR)/bundle.o $(C_BUILD_DIR)/allocate_free.o $(C_BUILD_DIR)/bab_functions.o \
	 	 $(C_BUILD_DIR)/bounding.o $(C_BUILD_DIR)/cutting_planes.o \
         $(C_BUILD_DIR)/evaluate.o $(C_BUILD_DIR)/heap.o $(C_BUILD_DIR)/ipm_mc_pk.o \
         $(C_BUILD_DIR)/heuristic.o $(C_BUILD_DIR)/main.o $(C_BUILD_DIR)/operators.o \
         $(C_BUILD_DIR)/process_input.o $(C_BUILD_DIR)/qap_simulated_annealing.o

# Python extension objects
OBJS =   $(WRAPPER_BUILD_DIR)/bundle.o $(WRAPPER_BUILD_DIR)/allocate_free.o $(WRAPPER_BUILD_DIR)/bab_functions.o \
	 	 $(WRAPPER_BUILD_DIR)/bounding.o $(WRAPPER_BUILD_DIR)/cutting_planes.o \
         $(WRAPPER_BUILD_DIR)/evaluate.o $(WRAPPER_BUILD_DIR)/heap.o $(WRAPPER_BUILD_DIR)/ipm_mc_pk.o \
         $(WRAPPER_BUILD_DIR)/heuristic.o $(WRAPPER_BUILD_DIR)/main.o $(WRAPPER_BUILD_DIR)/operators.o \
         $(WRAPPER_BUILD_DIR)/process_input.o $(WRAPPER_BUILD_DIR)/qap_simulated_annealing.o \
		 $(WRAPPER_BUILD_DIR)/wrapper_hooks.o $(WRAPPER_BUILD_DIR)/wrapper.o

CFLAGS = $(OPTI) -Wall -W -pedantic
CXXFLAGS = $(CPPOPTI) -Wall -W -pedantic

# Build rules
.PHONY: all rebuild clean clean-output clean-artifacts test tests test-unit test-harness \
        test-logging test-summaries \
        test-maxcut test-maxcut-python test-qubo-python test-qubo-qplib \
        test-qubo-python-heuristic test-bqp-python test-input-solution \
        test-parsers test-modular-combos test-expected-failure test-coverage-gaps \
        docker docker-dev docker-no-cache docker-clean docker-test docker-shell \
        lint format format-check coverage docs docs-serve

# Default rule is to build and install all binaries.
all: $(BIQBIN_BINARY) biqbin/$(PYMODULE)

# Force a complete rebuild; the nested make runs avoid parallel clean/build races.
rebuild:
	$(MAKE) clean
	$(MAKE) all

# Install native executable.
$(BIQBIN_BINARY): $(BINS)
	cp $< $@

# Install Python extension.
biqbin/$(PYMODULE): $(PYMOD_OUT)
	cp $< $@

clean-output:
	rm -f rudy/*.output*
	rm -f tests/rudy/*.output*
	rm -f tests/qubos/*/*.output*
	rm -f tests/bqp/*.output*
	rm -f tests/qplib/*.output*
	rm -f tests/*/*.output*

# Remove built outputs
clean: clean-output
	rm -rf build/
	rm -f $(BIQBIN_BINARY) biqbin/$(PYMODULE)

# Delete reports separately from native build products.
clean-artifacts:
	rm -rf $(ARTIFACT_DIR)

# Ensure output directories exist
$(WRAPPER_BUILD_DIR) $(C_BUILD_DIR):
	mkdir -p $@

# Rules for binaries
$(BINS): $(C_OBJS)
	$(CC) -o $@ $^ $(INCLUDES) $(LIB) $(CFLAGS) -DPURE_C $(LINALG)

$(C_BUILD_DIR)/%.o: src/%.c src/logging.h | $(C_BUILD_DIR)
	$(CC) $(CFLAGS) $(LOG_DEFINES) -DPURE_C $(INCLUDES) -MMD -MP -c -o $@ $<

# BiqBin code rules
$(WRAPPER_BUILD_DIR)/%.o: src/%.c src/logging.h | $(WRAPPER_BUILD_DIR)
	$(CC) $(CFLAGS) $(LOG_DEFINES) $(INCLUDES) -MMD -MP -c -o $@ $<

$(WRAPPER_BUILD_DIR)/%.o: src/%.cpp src/logging.h | $(WRAPPER_BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LOG_DEFINES) $(INCLUDES) -MMD -MP -c -o $@ $<

# Python module rule
$(PYMOD_OUT): $(OBJS)
	$(CXX) -o $@ $^ -shared -fPIC $(INCLUDES) $(LIB) $(LINALG) -Wl,--no-undefined

# Track header dependencies discovered by the compiler.
-include $(C_OBJS:.o=.d) $(OBJS:.o=.d)

# Every standalone test target requires a usable native build.
TEST_TARGETS := test-harness test-logging test-summaries test-maxcut \
    test-maxcut-python test-qubo-python test-qubo-qplib \
    test-qubo-python-heuristic test-bqp-python test-input-solution \
    test-parsers test-modular-combos test-expected-failure test-coverage-gaps
$(TEST_TARGETS): all

# Tests
# Create the JUnit directory for standalone and aggregate test targets.
$(TEST_TARGETS): | junit-output-dir
.PHONY: junit-output-dir
junit-output-dir:
	mkdir -p $(JUNIT_DIR) $(TEST_RESULTS_DIR)/behavior

test-harness:
	$(PYTHON) -m pytest tests/test_native_harness.py tests/test_solution_checks.py tests/test_behavior_report.py tests/test_makefile.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_native_harness.xml) -v --no-header

test-logging:
	$(RUN_ENVS) $(PYTHON) -m pytest tests/test_logging.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_logging.xml) -v --no-header

test-summaries:
	$(RUN_ENVS) $(PYTHON) -m pytest tests/test_solution_summary.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_solution_summary.xml) -v --no-header

test-maxcut: clean-output
	$(RUN_ENVS) tests/test.sh "$(MPI_RUN) ./$(BINS)" tests/rudy/g05_60.0 tests/rudy/g05_60.0-expected_output params
	$(RUN_ENVS) tests/test.sh "$(MPI_RUN) ./$(BINS)" tests/rudy/g05_80.0 tests/rudy/g05_80.0-expected_output params
	$(RUN_ENVS) tests/test.sh "$(MPI_RUN) ./$(BINS)" tests/rudy/g05_100.4 tests/rudy/g05_100.4-expected_output params

test-maxcut-python: clean-output
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_maxcut.py tests/rudy/g05_60.0.json -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_maxcut.py tests/rudy/g05_80.0.json -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_maxcut.py tests/rudy/g05_100.4.json -c

	$(PYTHON) -m pytest tests/test_biqbin_output.py --behavior-report=$(TEST_RESULTS_DIR)/behavior/$@.txt $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_biqbin_output.xml) -v -s --no-header --instances \
		tests/rudy/g05_60.0.json \
		tests/rudy/g05_80.0.json \
		tests/rudy/g05_100.4.json
	$(PYTHON) -m pytest tests/test_maxcut.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_maxcut.xml) -v -s --no-header

test-qubo-python: clean-output
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_qubo.py tests/qubos/80/kcluster80_025_20_1.json -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_qubo.py tests/qplib/5881.qplib --format=qplib -c
	$(PYTHON) -m pytest tests/test_biqbin_output.py --behavior-report=$(TEST_RESULTS_DIR)/behavior/$@.txt $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_biqbin_output.xml) -v -s --no-header --instances tests/qubos/40/kcluster40_025_10_1.json tests/qubos/80/kcluster80_025_20_1.json tests/qplib/5881.qplib
	$(PYTHON) -m pytest tests/test_qubo.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_qubo.xml) -v -s --no-header

test-qubo-qplib: clean-output
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_qubo.py tests/qplib/kcluster40_025_10_1.qplib --format=qplib -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_qubo.py tests/qplib/kcluster80_025_20_1.qplib --format=qplib -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_qubo.py tests/qplib/5881.qplib --format=qplib -c
	$(PYTHON) -m pytest tests/test_biqbin_output.py --behavior-report=$(TEST_RESULTS_DIR)/behavior/$@.txt $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_biqbin_output.xml) -v -s --no-header --instances tests/qplib/kcluster40_025_10_1.qplib tests/qplib/kcluster80_025_20_1.qplib tests/qplib/5881.qplib

test-qubo-python-heuristic: clean-output
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_heuristic.py tests/qubos/40/kcluster40_025_10_1.json \
				--output tests/heuristic/kcluster40_025_10_1.json.output.json -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_heuristic.py tests/qubos/80/kcluster80_025_20_1.json \
				--output tests/heuristic/kcluster80_025_20_1.json.output.json -c

	$(PYTHON) -m pytest tests/test_biqbin_output.py --behavior-report=$(TEST_RESULTS_DIR)/behavior/$@.txt $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_biqbin_output.xml) -v -s --no-header --instances tests/heuristic/kcluster40_025_10_1.json tests/heuristic/kcluster80_025_20_1.json

test-input-solution: clean-output
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_maxcut.py \
				tests/rudy/g05_60.0.json \
				--output tests/w_solution/g05_60.0.json.output.json \
				-s tests/w_solution/g05_60.0.json_initial_solution.json \
				-c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_qubo.py \
				tests/qubos/40/kcluster40_025_10_1.json \
				--output tests/w_solution/kcluster40_025_10_1.json.output.json \
				-s tests/w_solution/kcluster40_025_10_1.json_initial_solution.json \
				-c
	$(PYTHON) -m pytest tests/test_biqbin_output.py --behavior-report=$(TEST_RESULTS_DIR)/behavior/$@.txt $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_biqbin_output.xml) -v -s --no-header --instances tests/w_solution/g05_60.0.json tests/w_solution/kcluster40_025_10_1.json

test-bqp-python: clean-output
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_bqp.py tests/bqp/test_bqp.data -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_bqp.py tests/bqp/test_bqp.json -j -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_bqp.py tests/bqp/kcluster40_025_10_1_QBO_dense.json --format dense -c
	$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) biqbin_bqp.py tests/bqp/kcluster40_025_10_1_QBO_sparse.json --format sparse -c

	$(PYTHON) -m pytest tests/test_biqbin_output.py --behavior-report=$(TEST_RESULTS_DIR)/behavior/$@.txt $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_biqbin_output.xml) -v -s --no-header --instances tests/bqp/test_bqp.data tests/bqp/test_bqp.json tests/bqp/kcluster40_025_10_1_QBO_dense.json tests/bqp/kcluster40_025_10_1_QBO_sparse.json
	$(PYTHON) -m pytest tests/test_bqp.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_bqp.xml) -v -s --no-header

test-modular-combos: clean-output
	@set -e; \
	set --; \
	for case in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do \
		instance=tests/custom_solvers_results/small_example_qubo_test_case_$$case.json; \
		echo "Custom solver case $$case"; \
		$(RUN_ENVS) $(MPI_RUN) $(MPI_PYTHON) -m tests.biqbin_custom_solvers \
			tests/custom_solvers_results/small_example_qubo.json \
			-o "$$instance.output" --test-case $$case -c; \
		set -- "$$@" "$$instance"; \
	done; \
	$(PYTHON) -m pytest tests/test_biqbin_output.py --behavior-report=$(TEST_RESULTS_DIR)/behavior/$@.txt $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_biqbin_output.xml) -v -s --no-header --instances "$$@"

test-parsers:
	$(PYTHON) -m pytest tests/data_parsers $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-data_parsers.xml) -v --no-header

test-expected-failure:
	$(PYTHON) -m pytest tests/test_mpi_failure_cases.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-test_mpi_failure_cases.xml) -v -s

test-coverage-gaps:
	$(PYTHON) -m pytest \
		tests/test_bqp_initial_estimate.py \
		tests/test_time_limit_parsing.py \
		tests/test_utils_extra.py \
		tests/test_cli_output.py \
		tests/test_output_paths.py \
		$(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@.xml) \
		-v --no-header

test: all $(TEST_TARGETS)

tests: test

# ============================================================
# Docker
# ============================================================

docker:
	docker build $(DOCKER_BUILD_PARAMS) \
		--target runtime \
		--progress=plain \
		-t $(IMAGE):$(TAG) .

docker-dev:
	docker build $(DOCKER_BUILD_PARAMS) \
		--target ci \
		--progress=plain \
		-t $(IMAGE_DEV):$(TAG) .

docker-no-cache:
	docker build --no-cache $(DOCKER_BUILD_PARAMS) \
		--target runtime \
		--progress=plain \
		-t $(IMAGE):$(TAG) .

docker-clean:
	docker rmi -f $(IMAGE):$(TAG) || true
	docker rmi -f $(IMAGE_DEV):$(TAG) || true

docker-test: docker-dev
	docker run --rm --init \
		$(IMAGE_DEV):$(TAG) \
		make test

docker-shell: docker
	docker run --interactive --tty --rm \
		--mount type=bind,src=$(shell pwd)/$(DATA_DIR),dst=/data \
		$(IMAGE):$(TAG) /bin/bash

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

test-unit: all test-logging test-summaries test-parsers test-coverage-gaps
	$(RUN_ENVS) $(PYTHON) -m pytest tests/test_maxcut.py tests/test_qubo.py tests/test_bqp.py $(if $(JUNIT_DIR),--junitxml=$(JUNIT_DIR)/$@-solvers.xml) -v --no-header

# All pytest and MPI ranks inherit this database path.
coverage: export COVERAGE_FILE := $(abspath $(COVERAGE_DIR)/.coverage)
coverage:
	mkdir -p $(COVERAGE_DIR) $(TEST_RESULTS_DIR)/junit $(TEST_RESULTS_DIR)/behavior
	$(PYTHON) -m coverage erase
	@set +e; \
	PYTEST_ADDOPTS="$(PYTEST_ADDOPTS) --cov=biqbin --cov-branch --cov-append --cov-report=" \
		$(MAKE) test \
			JUNIT_DIR=$(TEST_RESULTS_DIR)/junit \
			MPI_PYTHON="$(PYTHON) -m coverage run --parallel-mode"; \
	status=$$?; \
	if find "$(COVERAGE_DIR)" -maxdepth 1 -name '.coverage.*' -print -quit | grep -q .; then \
		$(PYTHON) -m coverage combine --append || status=1; \
	fi; \
	if [ -f "$(COVERAGE_FILE)" ]; then \
		$(PYTHON) -m coverage report -m || status=1; \
		$(PYTHON) -m coverage xml -o "$(COVERAGE_DIR)/coverage.xml" || status=1; \
		$(PYTHON) -m coverage html -d "$(COVERAGE_DIR)/htmlcov" || status=1; \
	else \
		echo 'No combined coverage database was generated' >&2; \
		status=1; \
	fi; \
	exit $$status


docs:
	$(PYTHON) -m zensical build --strict

docs-serve:
	$(PYTHON) -m zensical serve
