all : main
	./main

main : main.cu
	nvcc -arch=compute_75 -o $@ $<
