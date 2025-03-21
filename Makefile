#### BiqBin makefile ####

# container image name
IMAGE ?= parallel-biqbin-maxcut
# container image tag
TAG ?= 1.0.0
DOCKER_BUILD_PARAMS ?=

# Directories
OBJ = Obj

# Compiler
CC = mpicc

LINALG 	 = -lopenblas -lm 
OPTI     = -O3 -ffast-math -fexceptions -fPIC -fno-common -g

# binary
BINS =  biqbin

# test command
NUM_PROC = 8
PARAMS = test/params
TEST_INSTANCE = test/Instances/rudy/g05_60.4
TEST_EXPECTED = test/Instances/rudy/g05_60.4-expected_output

TEST = ./test.sh \
	"mpiexec -n $(NUM_PROC) ./$(BINS)" \
	$(TEST_INSTANCE) \
	$(TEST_EXPECTED) \
	test/params

TEST_ALL_60 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) ./$(BINS)" \
			test/Instances/rudy/g05_60.$$i \
			test/Instances/rudy/g05_60.$$i-expected_output \
			params ;\
	done

TEST_ALL_80 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) ./$(BINS)" \
			test/Instances/rudy/g05_80.$$i \
			test/Instances/rudy/g05_80.$$i-expected_output \
			params ;\
	done

TEST_ALL_100 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) ./$(BINS)" \
			test/Instances/rudy/g05_100.$$i \
			test/Instances/rudy/g05_100.$$i-expected_output \
			params ;\
	done

TEST_PYTHON = ./test.sh \
	"mpiexec -n $(NUM_PROC) python3 test.py" \
	test/Instances/rudy/g05_60.0 \
	test/Instances/rudy/g05_60.0-expected_output \
	test/params

TEST_ALL_PYTHON_60 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) python3 test.py" \
			test/Instances/rudy/g05_60.$$i \
			test/Instances/rudy/g05_60.$$i-expected_output \
			params ;\
	done

TEST_ALL_PYTHON_80 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) python3 test.py" \
			test/Instances/rudy/g05_80.$$i \
			test/Instances/rudy/g05_80.$$i-expected_output \
			params ;\
	done

TEST_ALL_PYTHON_100 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) python3 test.py" \
			test/Instances/rudy/g05_100.$$i \
			test/Instances/rudy/g05_100.$$i-expected_output \
			params ;\
	done


TEST_P = ./test.sh \
	"mpiexec -n $(NUM_PROC) python3 mpi_test.py" \
	test/Instances/rudy/g05_60.0 \
	test/Instances/rudy/g05_60.0-expected_output \
	test/params

TEST_ALL_P_60 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) python3 mpi_test.py" \
			test/Instances/rudy/g05_60.$$i \
			test/Instances/rudy/g05_60.$$i-expected_output \
			params ;\
	done

TEST_ALL_P_80 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) python3 mpi_test.py" \
			test/Instances/rudy/g05_80.$$i \
			test/Instances/rudy/g05_80.$$i-expected_output \
			params ;\
	done

TEST_ALL_P_100 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec -n $(NUM_PROC) python3 mpi_test.py" \
			test/Instances/rudy/g05_100.$$i \
			test/Instances/rudy/g05_100.$$i-expected_output \
			params ;\
	done


# BiqBin objects
BBOBJS = $(OBJ)/bundle.o $(OBJ)/allocate_free.o $(OBJ)/bab_functions.o \
		 $(OBJ)/bounding.o $(OBJ)/cutting_planes.o \
         $(OBJ)/evaluate.o $(OBJ)/heap.o $(OBJ)/ipm_mc_pk.o \
         $(OBJ)/heuristic.o $(OBJ)/main.o $(OBJ)/operators.o \
         $(OBJ)/process_input.o $(OBJ)/qap_simulated_annealing.o \
		 $(OBJ)/biqbin.o $(OBJ)/wrapper_functions.o


# All objects
OBJS = $(BBOBJS)
CFLAGS = $(OPTI) -Wall -W -pedantic 


#### Rules ####

.PHONY : all clean

# Default rule is to create all binaries #
all: $(BINS) biqbin.so

test: all
	$(TEST)
	$(TEST_PYTHON)

# test-memory:
# 	mpiexec -n 8 valgrind --leak-check=full --show-leak-kinds=all ./$(BINS) $(TEST_INSTANCE) $(PARAMS)

test-all:
	$(TEST_ALL_60)
	$(TEST_ALL_PYTHON_60)
	$(TEST_ALL_80)
	$(TEST_ALL_PYTHON_80)
	$(TEST_ALL_100)
	$(TEST_ALL_PYTHON_100)

test-all-python:
	$(TEST_ALL_PYTHON_60)
	$(TEST_ALL_PYTHON_80)
	$(TEST_ALL_PYTHON_100)

test-all-p:
	$(TEST_ALL_P_60)
	$(TEST_ALL_P_80)
	$(TEST_ALL_P_100)

run:
	mpirun -n 8 ./$(BINS) \
	$(TEST_INSTANCE) \
	$(PARAMS)

run-all-60:
	for i in $(shell seq 0 9); do \
			mpiexec -n 8 ./$(BINS) \
			test/Instances/rudy/g05_60.$$i \
			params ;\
	done

run-all-80:
	for i in $(shell seq 0 9); do \
			mpiexec -n 8 ./$(BINS) \
			test/Instances/rudy/g05_80.$$i \
			params ;\
	done

run-all-100:
	for i in $(shell seq 0 9); do \
			mpiexec -n 8 ./$(BINS) \
			test/Instances/rudy/g05_100.$$i \
			params ;\
	done

run-python:
	mpirun -n 8 python3 test.py \
	$(TEST_INSTANCE) \
	$(PARAMS)

run-python-all-60:
	for i in $(shell seq 0 9); do \
		mpiexec -n 8 python3 test.py \
		test/Instances/rudy/g05_60.$$i \
		$(PARAMS); \
	done

run-python-all-80:
	for i in $(shell seq 0 9); do \
		mpiexec -n 8 python3 test.py \
		test/Instances/rudy/g05_80.$$i \
		$(PARAMS); \
	done


run-python-all-100:
	for i in $(shell seq 0 9); do \
		mpiexec -n 8 python3 test.py \
		test/Instances/rudy/g05_100.$$i \
		$(PARAMS); \
	done

docker: 
	docker build $(DOCKER_BUILD_PARAMS) --progress=plain -t $(IMAGE):$(TAG)  . 

docker-test:
	docker run --rm $(IMAGE):$(TAG) $(TEST)
	
docker-clean: 
	docker rmi -f $(IMAGE):$(TAG) 


# Rules for binaries #
$(BINS) : $(OBJS)
	$(CC) -o $@ $^ $(INCLUDES) $(LIB) $(OPTI) $(LINALG)  


# BiqBin code rules 
$(OBJ)/%.o : %.c | $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
		
$(OBJ):
	mkdir -p $(OBJ)

biqbin.so: $(OBJS)
	$(CC) -shared -o biqbin.so $(OBJS) $(LINALG)

# Clean rule #
clean :
	rm -rf $(BINS) $(OBJ)
	rm -f biqbin.so
	rm -rf __pycache__

clean-output:
	rm -rf test/Instances/rudy/*.output*
	rm -rf Instances/rudy/*.output*