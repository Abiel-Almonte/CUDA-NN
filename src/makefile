INCLUDES = -I./layers -I./utils

SRCS = $(wildcard ./*.cu)   $(wildcard ./layers/*.cu)   $(wildcard ./utils/*.cu)
OBJS = $(SRCS:.cu=.o)

EXEC = cudaNN

$(EXEC): $(OBJS)
	nvcc $(INCLUDES) $^ -o $@

%.o: %.cu
	nvcc -c $(INCLUDES) $^ -o $@

clean:
	rm -f $(OBJS) $(EXEC)