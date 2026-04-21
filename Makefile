CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -pthread -Wall -Wextra

all: fs_metadata_bench

fs_metadata_bench: fs_metadata_bench.cpp
	$(CXX) $(CXXFLAGS) fs_metadata_bench.cpp -o fs_metadata_bench

clean:
	rm -f fs_metadata_bench
