CXX=g++

all: bench_cos bench_pop

bench_cos : cosine.cpp
	$(CXX) -o bench_cos cosine.cpp -O3 -lbenchmark_main -std=c++17 -I/usr/local/include -L/usr/local/lib -lbenchmark

bench_pop : pop.cpp
	$(CXX) -o bench_pop pop.cpp -march=tigerlake -O3 -lbenchmark_main -std=c++17 -I/usr/local/include -L/usr/local/lib -lbenchmark

clean:
	rm -rf bench_pop bench_cos
