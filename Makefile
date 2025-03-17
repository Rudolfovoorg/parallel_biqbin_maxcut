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
OPTI     = -O3 -ffast-math -fexceptions -fPIC -fno-common

# binary
BINS =  biqbin

# test command
TEST = ./test.sh \
	./$(BINS) \
	test/Instances/rudy/g05_60.0 \
	test/Instances/rudy/g05_60.0-expected_output \
	test/params

TEST_ALL_60 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			./$(BINS) \
			test/Instances/rudy/g05_60.$$i \
			test/Instances/rudy/g05_60.$$i-expected_output \
			params ;\
	done

TEST_ALL_80 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			./$(BINS) \
			test/Instances/rudy/g05_80.$$i \
			test/Instances/rudy/g05_80.$$i-expected_output \
			params ;\
	done

TEST_ALL_100 = 	for i in $(shell seq 0 9); do \
			./test.sh \
			./$(BINS) \
			test/Instances/rudy/g05_100.$$i \
			test/Instances/rudy/g05_100.$$i-expected_output \
			params ;\
	done


# python test command
# TEST_PYTHON = ./test.sh \
# 	"python3 test.py" \
# 	test/Instances/rudy/g05_60.0 \
# 	test/Instances/rudy/g05_60.0-expected_output \
# 	test/params

# BiqBin objects
BBOBJS = $(OBJ)/bundle.o $(OBJ)/allocate_free.o $(OBJ)/bab_functions.o \
	 $(OBJ)/bounding.o $(OBJ)/cutting_planes.o \
         $(OBJ)/evaluate.o $(OBJ)/heap.o $(OBJ)/ipm_mc_pk.o \
         $(OBJ)/heuristic.o $(OBJ)/main.o $(OBJ)/operators.o \
         $(OBJ)/process_input.o $(OBJ)/qap_simulated_annealing.o

# All objects
OBJS = $(BBOBJS)

CFLAGS = $(OPTI) -Wall -W -pedantic 


#### Rules ####

.PHONY : all clean

# Default rule is to create all binaries #
all: $(BINS)

test: all
	$(TEST)

test-all:
	$(TEST_ALL_60)
	$(TEST_ALL_80)
	$(TEST_ALL_100)

run-all-100:
	for i in $(shell seq 0 9); do \
			mpiexec -n 8 ./$(BINS) \
			test/Instances/rudy/g05_100.$$i \
			params ;\
	done

run-all-80:
	for i in $(shell seq 0 9); do \
			mpiexec -n 8 ./$(BINS) \
			test/Instances/rudy/g05_80.$$i \
			params ;\
	done

run-all-60:
	for i in $(shell seq 0 9); do \
			mpiexec -n 8 ./$(BINS) \
			test/Instances/rudy/g05_60.$$i \
			params ;\
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

# Clean rule #
clean :
	rm -rf $(BINS) $(OBJS) $(OBJ)

clean-output:
	rm -rf test/Instances/rudy/*.output*
	rm -rf Instances/rudy/*.output*
	rm -rf .output*
