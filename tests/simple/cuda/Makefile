include ../../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

all: simple

simple: simple.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f simple
