CXX=g++
bench : main.cpp
	$(CXX) -o bench main.cpp -march=tigerlake -O3 -lbenchmark_main -std=c++17 -I/usr/local/include -L/usr/local/lib -lbenchmark
