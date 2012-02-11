include ../../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

all: nested

nested: nested.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f nested
