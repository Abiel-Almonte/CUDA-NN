INCLUDES = -I./layers -I./utils
CXXFLAGS = -O2
ARCH = compute_80 
CODE = sm_80
SRCS = $(wildcard ./*.cu)   $(wildcard ./layers/*.cu)   $(wildcard ./utils/*.cu)
OBJS = $(SRCS:.cu=.o)

EXEC = cudaNN

$(EXEC): $(OBJS) 
	nvcc $(INCLUDES) $^ -o $@ -arch=$(ARCH) -code=$(CODE)

%.o: %.cu
	nvcc -c $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -f $(OBJS) $(EXEC)