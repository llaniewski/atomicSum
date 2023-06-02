all : main
	./main

main : main.cu
	nvcc -o $@ $<
