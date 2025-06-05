#### BiqBin makefile ####

# container image name
IMAGE ?= parallel-biqbin
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
CPPOPTI  = -O3 -fexceptions -fPIC -fno-common -ffast-math

# Python module (Boost)
PYBOOST_INCLUDES ?= -I$(CONDA_PREFIX)/include/python3.12 -I$(CONDA_PREFIX)/include
PYBOOST_LIBS ?= -L$(CONDA_PREFIX)/lib -lboost_python312 -lboost_numpy312 -lpython3.12

INCLUDES += $(PYBOOST_INCLUDES)
LIB += $(PYBOOST_LIBS)

# Python module (Boost)
PYMODULE = biqbin.so
PYMOD_OUT = $(WRAPPER_BUILD_DIR)/$(PYMODULE)
# C only binary
BIQBIN_BINARY = biqbin
BINS =  $(C_BUILD_DIR)/$(BIQBIN_BINARY)

RUN_ENVS = OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1

# BiqBin objects
C_OBJS = $(C_BUILD_DIR)/bundle.o $(C_BUILD_DIR)/allocate_free.o $(C_BUILD_DIR)/bab_functions.o \
	 	 $(C_BUILD_DIR)/bounding.o $(C_BUILD_DIR)/cutting_planes.o \
         $(C_BUILD_DIR)/evaluate.o $(C_BUILD_DIR)/heap.o $(C_BUILD_DIR)/ipm_mc_pk.o \
         $(C_BUILD_DIR)/heuristic.o $(C_BUILD_DIR)/main.o $(C_BUILD_DIR)/operators.o \
         $(C_BUILD_DIR)/process_input.o $(C_BUILD_DIR)/qap_simulated_annealing.o \
		 $(C_BUILD_DIR)/bqp_data_processing.o

# BiqBin objects
OBJS =   $(WRAPPER_BUILD_DIR)/bundle.o $(WRAPPER_BUILD_DIR)/allocate_free.o $(WRAPPER_BUILD_DIR)/bab_functions.o \
	 	 $(WRAPPER_BUILD_DIR)/bounding.o $(WRAPPER_BUILD_DIR)/cutting_planes.o \
         $(WRAPPER_BUILD_DIR)/evaluate.o $(WRAPPER_BUILD_DIR)/heap.o $(WRAPPER_BUILD_DIR)/ipm_mc_pk.o \
         $(WRAPPER_BUILD_DIR)/heuristic.o $(WRAPPER_BUILD_DIR)/main.o $(WRAPPER_BUILD_DIR)/operators.o \
         $(WRAPPER_BUILD_DIR)/process_input.o $(WRAPPER_BUILD_DIR)/qap_simulated_annealing.o \
		 $(WRAPPER_BUILD_DIR)/bqp_data_processing.o

# All objects

CFLAGS = $(OPTI) -Wall -W -pedantic 
CPPFLAGS = $(CPPOPTI) -Wall -W -pedantic 

#### Rules ####

.PHONY : all clean test tests

# Default rule is to create all binaries #
all: clean $(BINS) $(PYMOD_OUT)
	cp $(PYMOD_OUT) .
	cp $(BINS) .

	
clean-output:
	rm -f rudy/*.output*
	rm -f tests/rudy/*.output*
	rm -f tests/qubos/*/*.output*
	rm -f evil_qubos/*.output*

# Clean rule #
clean: clean-output
	rm -rf build/
	rm -rf $(BIQBIN_BINARY)
	rm -rf $(PYMODULE)

# Ensure output directories exist
$(WRAPPER_BUILD_DIR) $(C_BUILD_DIR):
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

$(WRAPPER_BUILD_DIR)/wrapper.o: src/wrapper.cpp  | $(WRAPPER_BUILD_DIR)
	$(CPP) $(CPPFLAGS) $(INCLUDES) -c -o $@ $<

# Python module rule
$(PYMOD_OUT): $(OBJS) build/wrapper/wrapper.o
	$(CPP) -o $@ $^ -shared -fPIC $(INCLUDES) $(LIB) $(LINALG) -Wl,--no-undefined

# Tests
test-maxcut: clean-output
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 ./$(BINS)" tests/rudy/g05_60.0 tests/rudy/g05_60.0-expected_output params
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 ./$(BINS)" tests/rudy/g05_80.0 tests/rudy/g05_80.0-expected_output params
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 ./$(BINS)" tests/rudy/g05_100.4 tests/rudy/g05_100.4-expected_output params

test-maxcut-python: clean-output
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 python biqbin_maxcut.py" tests/rudy/g05_60.0 tests/rudy/g05_60.0-expected_output params
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 python biqbin_maxcut.py" tests/rudy/g05_80.0 tests/rudy/g05_80.0-expected_output params
	$(RUN_ENVS) tests/test.sh "mpiexec -n 3 python biqbin_maxcut.py" tests/rudy/g05_100.4 tests/rudy/g05_100.4-expected_output params

test-qubo-python: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qubos/40/kcluster40_025_10_1.json params
	python tests/check_qubo_test.py tests/qubos/40/kcluster40_025_10_1.json
	$(RUN_ENVS) mpiexec -n 3 python biqbin_qubo.py tests/qubos/80/kcluster80_025_20_1.json params
	python tests/check_qubo_test.py tests/qubos/80/kcluster80_025_20_1.json

test-qubo-python-heuristic: clean-output
	$(RUN_ENVS) mpiexec -n 3 python biqbin_heuristic.py tests/qubos/40/kcluster40_025_10_1.json params
	python tests/check_qubo_test.py tests/qubos/40/kcluster40_025_10_1.json
	$(RUN_ENVS) mpiexec -n 3 python biqbin_heuristic.py tests/qubos/80/kcluster80_025_20_1.json params
	python tests/check_qubo_test.py tests/qubos/80/kcluster80_025_20_1.json

test: test-maxcut test-maxcut-python test-qubo-python test-qubo-python-heuristic

docker: 
	docker build $(DOCKER_BUILD_PARAMS) --progress=plain -t $(IMAGE):$(TAG)  . 

docker-no-cache: 
	docker build --no-cache $(DOCKER_BUILD_PARAMS) --progress=plain -t $(IMAGE):$(TAG)  . 

docker-clean: 
	docker rmi -f $(IMAGE):$(TAG) 

docker-test:
	docker run --rm $(IMAGE):$(TAG) make test

docker-shell:
	docker run --interactive --tty --rm --mount type=bind,src=$(shell pwd)/$(DATA_DIR),dst=/data $(IMAGE):$(TAG) /bin/bash

