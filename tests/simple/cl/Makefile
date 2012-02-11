include ../../../Makefile.common
TOP := $(dir $(lastword $(MAKEFILE_LIST)))
INC = -I $(TOP)/include
LIB = -L$(TOP)/lib -lclwrapper

all: simple

simple: simple.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) $(INC) -o $@ $< $(LIB)

clean:
	rm -f simple
