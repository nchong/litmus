include ../../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))

nested: nested.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f nested
