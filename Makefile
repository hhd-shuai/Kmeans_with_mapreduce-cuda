NVCC = nvcc
CXX = g++ -std=c++11

TARGETS = kmeans_with_mapreduce-cuda
SOURCES = kmeans_with_mapreduce-cuda.cu kmeans_mapreduce_core.cu
DEF_HEADER = random_num_generator.hpp
HEADERS = config.cuh $(DEF_HEADER)
OBJECTS = $(patsubst %.cu,%.o,$(SOURCES))

kmeans_with_mapreduce-cuda: $(SOURCES) $(HEADERS)
	$(NVCC) -std=c++11 -O3 -dc $(SOURCES)
	$(NVCC) -std=c++11 -O3 -o $@ $(OBJECTS)

clean:
	rm -f *.o $(TARGETS)