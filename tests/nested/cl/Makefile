include ../../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))
INC = -I $(TOP)/include
LIB = -L$(TOP)/lib -lclwrapper

nested: nested.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) $(INC) -o $@ $< $(LIB)

clean:
	rm -f nested
