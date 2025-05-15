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
OPTI     = -O3 -ffast-math -fexceptions -fPIC -fno-common -shared

# binary
BINS =  biqbin

# test command
PARAMS = test/params
TEST_INSTANCE = test/Instances/rudy/g05_60.0
TEST_EXPECTED = $(TEST_INSTANCE)-expected_output

TEST = ./test.sh \
	"mpiexec python3 run_example.py" \
	$(TEST_INSTANCE) \
	$(TEST_EXPECTED) \
	test/params

TEST_HUGE = ./test.sh \
	"mpiexec python3 run_example.py" \
	test/Instances/rudy/be250.10 \
	test/Instances/rudy/be250.10-expected_output \
	test/params

TEST_ALL_60 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec python3 run_example.py" \
			test/Instances/rudy/g05_60.$$i \
			test/Instances/rudy/g05_60.$$i-expected_output \
			params ;\
	done

TEST_ALL_80 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec python3 run_example.py" \
			test/Instances/rudy/g05_80.$$i \
			test/Instances/rudy/g05_80.$$i-expected_output \
			params ;\
	done

TEST_ALL_100 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			"mpiexec python3 run_example.py" \
			test/Instances/rudy/g05_100.$$i \
			test/Instances/rudy/g05_100.$$i-expected_output \
			params ;\
	done




# BiqBin objects
BBOBJS = $(OBJ)/bundle.o $(OBJ)/allocate_free.o \
		 $(OBJ)/cutting_planes.o $(OBJ)/bab_functions.o \
         $(OBJ)/heap.o $(OBJ)/ipm_mc_pk.o $(OBJ)/heuristic.o \
         $(OBJ)/wrapped_heuristics.o $(OBJ)/operators.o \
         $(OBJ)/process_input.o $(OBJ)/qap_simulated_annealing.o \
		 $(OBJ)/wrapper_functions.o $(OBJ)/wrapped_bounding.o \


# All objects
OBJS = $(BBOBJS)
CFLAGS = $(OPTI) -Wall -W -pedantic -Wextra


#### Rules ####

.PHONY : all clean

# Default rule is to create all binaries #
all: $(BINS) biqbin.so

test: all
	$(TEST)

test-memory:
	mpirun valgrind -s --show-leak-kinds=all --leak-check=full --log-file=valgrind.%p.log python3 run_example.py $(TEST_INSTANCE) test/params

test-all:
	$(TEST_ALL_60)
	$(TEST_ALL_80)
	$(TEST_ALL_100)

test-all-80:
	$(TEST_ALL_80)

test-all-100:
	$(TEST_ALL_100)

run-huge:
	mpirun python3 run_example.py test/Instances/rudy/be250.10 test/params

test-huge:
	$(TEST_HUGE)

run:
	mpirun python3 run_example.py \
	$(TEST_INSTANCE) \
	$(PARAMS)

run-all-60:
	for i in $(shell seq 0 9); do \
		mpiexec python3 run_example.py \
		test/Instances/rudy/g05_60.$$i \
		$(PARAMS); \
	done

run-all-80:
	for i in $(shell seq 0 9); do \
		mpiexec python3 run_example.py \
		test/Instances/rudy/g05_80.$$i \
		$(PARAMS); \
	done


run-all-100:
	for i in $(shell seq 0 9); do \
		mpiexec python3 run_example.py \
		test/Instances/rudy/g05_100.$$i \
		$(PARAMS); \
	done

docker:
	docker build $(DOCKER_BUILD_PARAMS) --progress=plain -t $(IMAGE):$(TAG)  . 

# docker-run:
# 	docker run -it -v ./ parallel-biqbin-maxcut:1.0.0 $(IMAGE):$(TAG)

docker-run:
	docker run -it \
	  -v "$(shell pwd):/solver" \
	  --name parallel-biqbin-container \
	  parallel-biqbin-maxcut:1.0.0 \
	  /bin/bash

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