#### BiqBin makefile ####

# container image name
IMAGE ?= parallel-biqbin
IMAGE_DEV ?= parallel-biqbin-dev
# container image tag
TAG ?= 1.0.0
DOCKER_BUILD_PARAMS ?=
DATA_DIR ?= tests/

# Directories
WRAPPER_BUILD_DIR = build/wrapper
C_BUILD_DIR = build/c_build

# Compiler
CC = mpicc
CPP = mpic++

LINALG 	 = -lopenblas -lm 
OPTI     = -O3 -ffast-math -fexceptions -fPIC -fno-common
CPPOPTI  = -O3 -ffast-math -fexceptions -fPIC -fno-common

PYTHON_VERSION := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE := $(shell python3-config --includes)
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTHON_LDFLAGS := $(shell python3-config --ldflags)
PYTHON_LIBS := $(shell python3-config --libs)

# Add explicit python lib
PYTHON_LIB := -lpython$(PYTHON_VERSION) $(PYTHON_LDFLAGS) $(PYTHON_LIBS)

INCLUDES += $(PYBIND11_INCLUDES) $(PYTHON_INCLUDE)
LIB += $(PYTHON_LIB)

# Python module (Pybind11)
PYMODULE = biqbin_module.so
PYMOD_OUT = $(WRAPPER_BUILD_DIR)/$(PYMODULE)
# C only binary
BIQBIN_BINARY = biqbin_executable
BINS =  $(C_BUILD_DIR)/$(BIQBIN_BINARY)

# BQP module (Pybind11)
BQP_BUILD_DIR = build/bqp_PLACEHOLDER
BQPMODULE = bqp_data_processing_PLACEHOLDER.so
BQPMOD_OUT = $(BQP_BUILD_DIR)/$(BQPMODULE)

RUN_ENVS = OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1

# BiqBin objects
C_OBJS = $(C_BUILD_DIR)/bundle.o $(C_BUILD_DIR)/allocate_free.o $(C_BUILD_DIR)/bab_functions.o \
	 	 $(C_BUILD_DIR)/bounding.o $(C_BUILD_DIR)/cutting_planes.o \
         $(C_BUILD_DIR)/evaluate.o $(C_BUILD_DIR)/heap.o $(C_BUILD_DIR)/ipm_mc_pk.o \
         $(C_BUILD_DIR)/heuristic.o $(C_BUILD_DIR)/main.o $(C_BUILD_DIR)/operators.o \
         $(C_BUILD_DIR)/process_input.o $(C_BUILD_DIR)/qap_simulated_annealing.o \

# BiqBin objects
OBJS =   $(WRAPPER_BUILD_DIR)/bundle.o $(WRAPPER_BUILD_DIR)/allocate_free.o $(WRAPPER_BUILD_DIR)/bab_functions.o \
	 	 $(WRAPPER_BUILD_DIR)/bounding.o $(WRAPPER_BUILD_DIR)/cutting_planes.o \
         $(WRAPPER_BUILD_DIR)/evaluate.o $(WRAPPER_BUILD_DIR)/heap.o $(WRAPPER_BUILD_DIR)/ipm_mc_pk.o \
         $(WRAPPER_BUILD_DIR)/heuristic.o $(WRAPPER_BUILD_DIR)/main.o $(WRAPPER_BUILD_DIR)/operators.o \
         $(WRAPPER_BUILD_DIR)/process_input.o $(WRAPPER_BUILD_DIR)/qap_simulated_annealing.o \
		 $(WRAPPER_BUILD_DIR)/wrapper_hooks.o $(WRAPPER_BUILD_DIR)/wrapper.o

# All objects

CFLAGS = $(OPTI) -Wall -W -pedantic 
CPPFLAGS = $(CPPOPTI) -Wall -W -pedantic 

#### Rules ####

.PHONY : all clean test tests

# Default rule is to create all binaries #
all: clean $(BINS) $(PYMOD_OUT) $(BQPMOD_OUT)
	cp $(PYMOD_OUT) biqbin/
	cp $(BINS) .
	cp $(BQPMOD_OUT) biqbin/

	
clean-output:
	rm -f rudy/*.output*
	rm -f tests/rudy/*.output*
	rm -f tests/qubos/*/*.output*
	rm -f tests/bqp/*.output*
	rm -f tests/qplib/*.output*
	rm -f tests/*/*.output*

# Clean rule #
clean: clean-output
	rm -rf build/
	rm -rf $(BIQBIN_BINARY)
	rm -rf biqbin/$(PYMODULE)
	rm -rf biqbin/$(BQPMODULE)

# Ensure output directories exist
$(WRAPPER_BUILD_DIR) $(C_BUILD_DIR) $(BQP_BUILD_DIR):
	mkdir -p build
	mkdir -p $@

# Rules for binaries
$(BINS): $(C_OBJS)
	$(CC) -o $@ $^ $(INCLUDES) $(LIB) $(CFLAGS) -DPURE_C $(LINALG)

$(C_BUILD_DIR)/%.o: src/%.c  | $(C_BUILD_DIR)
	$(CC) $(CFLAGS) -DPURE_C $(INCLUDES) -c -o $@ $<

# BiqBin code rules
$(WRAPPER_BUILD_DIR)/%.o: src/%.c  | $(WRAPPER_BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(WRAPPER_BUILD_DIR)/%.o: src/%.cpp  | $(WRAPPER_BUILD_DIR)
	$(CPP) $(CPPFLAGS) $(INCLUDES) -c -o $@ $<

# Python module rule
$(PYMOD_OUT): $(OBJS)
	$(CPP) -o $@ $^ -shared -fPIC $(INCLUDES) $(LIB) $(LINALG) -Wl,--no-undefined

# bqp module build
$(BQP_BUILD_DIR)/bqp_data_processing.o: src/bqp_PLACEHOLDER/bqp_data_processing.cpp | $(BQP_BUILD_DIR)
	$(CPP) $(CPPFLAGS) $(INCLUDES) -c -o $@ $<

$(BQPMOD_OUT): $(BQP_BUILD_DIR)/bqp_data_processing.o | $(BQP_BUILD_DIR)
	$(CPP) -o $@ $^ -shared -fPIC $(INCLUDES) $(LIB) $(LINALG) -Wl,--no-undefined

# Tests
test-maxcut: clean-output
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 ./$(BINS)" tests/rudy/g05_60.0 tests/rudy/g05_60.0-expected_output params
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 ./$(BINS)" tests/rudy/g05_80.0 tests/rudy/g05_80.0-expected_output params
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 ./$(BINS)" tests/rudy/g05_100.4 tests/rudy/g05_100.4-expected_output params

test-maxcut-python: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_maxcut.py tests/rudy/g05_60.0.json -c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_maxcut.py tests/rudy/g05_80.0.json -c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_maxcut.py tests/rudy/g05_100.4.json -c > /dev/null

	python -m pytest tests/test_biqbin_output.py -v -s --no-header --instances \
		tests/rudy/g05_60.0.json \
		tests/rudy/g05_80.0.json \
		tests/rudy/g05_100.4.json

test-qubo-python: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json -c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qubos/80/kcluster80_025_20_1.json -c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qplib/5881.qplib --format=qplib -c > /dev/null
	python -m pytest tests/test_biqbin_output.py -v -s --no-header --instances tests/qubos/40/kcluster40_025_10_1.json tests/qubos/80/kcluster80_025_20_1.json tests/qplib/5881.qplib

test-qubo-qplib: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qplib/kcluster40_025_10_1.qplib --format=qplib -c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qplib/kcluster80_025_20_1.qplib --format=qplib -c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qplib/5881.qplib --format=qplib -c > /dev/null
	python -m pytest tests/test_biqbin_output.py -v -s --no-header --instances tests/qplib/kcluster40_025_10_1.qplib tests/qplib/kcluster80_025_20_1.qplib tests/qplib/5881.qplib

test-qubo-python-heuristic: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_heuristic.py tests/qubos/40/kcluster40_025_10_1.json \
				--output tests/heuristic/kcluster40_025_10_1.json.output.json -c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_heuristic.py tests/qubos/80/kcluster80_025_20_1.json \
				--output tests/heuristic/kcluster80_025_20_1.json.output.json -c > /dev/null
	
	python -m pytest tests/test_biqbin_output.py -v -s --no-header --without-sol-vector --instances tests/heuristic/kcluster40_025_10_1.json tests/heuristic/kcluster80_025_20_1.json

test-input-solution: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_maxcut.py \
				tests/rudy/g05_60.0.json \
				--output tests/w_solution/g05_60.0.json.output.json \
				-s tests/w_solution/g05_60.0.json_initial_solution.json \
				-c > /dev/null
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py \
				tests/qubos/40/kcluster40_025_10_1.json \
				--output tests/w_solution/kcluster40_025_10_1.json.output.json \
				-s tests/w_solution/kcluster40_025_10_1.json_initial_solution.json \
				-c > /dev/null
	python -m pytest tests/test_biqbin_output.py -v -s --no-header --instances tests/w_solution/g05_60.0.json tests/w_solution/kcluster40_025_10_1.json

test-bqp-python: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_bqp.py tests/bqp/test_bqp.data -c > /dev/null 2>&1
	$(RUN_ENVS) mpiexec -n 3 python biqbin_bqp.py tests/bqp/test_bqp.json -j -c > /dev/null 2>&1
	python -m pytest tests/test_biqbin_output.py -v -s --no-header --instances tests/bqp/test_bqp.data tests/bqp/test_bqp.json

test-modular-combos: clean-output
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_0.json.output --test-case 0 -c > /dev/null 2>&1 
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_1.json.output --test-case 1 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_2.json.output --test-case 2 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_3.json.output --test-case 3 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_4.json.output --test-case 4 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_5.json.output --test-case 5 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_6.json.output --test-case 6 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_7.json.output --test-case 7 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_8.json.output --test-case 8 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_9.json.output --test-case 9 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_10.json.output --test-case 10 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_11.json.output --test-case 11 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_12.json.output --test-case 12 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_13.json.output --test-case 13 -c > /dev/null 2>&1
	$(RUN_ENVS) mpirun -n 3 python3 tests/biqbin_custom_solvers.py tests/custom_solvers_results/small_example_qubo.json -o tests/custom_solvers_results/small_example_qubo_test_case_14.json.output --test-case 14 -c > /dev/null 2>&1

	python -m pytest tests/test_biqbin_output.py -v -s --no-header --instances  tests/custom_solvers_results/small_example_qubo_test_case_0.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_1.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_2.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_3.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_4.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_5.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_6.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_7.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_8.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_9.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_10.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_11.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_12.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_13.json \
																				tests/custom_solvers_results/small_example_qubo_test_case_14.json \

test-parsers:
	python -m pytest tests/test_data_parsers.py -v --no-header


test: test-maxcut test-maxcut-python test-qubo-python test-qubo-python-heuristic test-bqp-python test-input-solution test-parsers test-modular-combos

docker: 
	docker build $(DOCKER_BUILD_PARAMS) --progress=plain -t $(IMAGE):$(TAG)  . 

docker-dev:
	docker build $(DOCKER_BUILD_PARAMS) --progress=plain -t $(IMAGE):$(TAG)  . 
	docker build $(DOCKER_BUILD_PARAMS) -f Dockerfile.dev --build-arg BASE_IMAGE=$(IMAGE):$(TAG) --progress=plain -t $(IMAGE_DEV):$(TAG)  . 

docker-no-cache: 
	docker build --no-cache $(DOCKER_BUILD_PARAMS) --progress=plain -t $(IMAGE):$(TAG)  . 

docker-clean: 
	docker rmi -f $(IMAGE):$(TAG)
	docker rmi -f $(IMAGE_DEV):$(TAG)

docker-test:
	docker run --rm $(IMAGE):$(TAG) sh -c 'pip install -r requirements-dev.txt && make test'

docker-shell:
	docker run --interactive --tty --rm --mount type=bind,src=$(shell pwd)/$(DATA_DIR),dst=/data $(IMAGE):$(TAG) /bin/bash

