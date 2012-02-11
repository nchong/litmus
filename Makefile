include Makefile.common

# DIRECTORIES
SRC = clwrapper
INC = include
BUILD = clwrapper/build
LIB = lib
OUT = $(LIB)/libclwrapper.so

# FINALLY THE RULES
all: $(OUT)

OBJS = $(addprefix $(BUILD)/,log.o clerror.o clwrapper.o)
$(OBJS): | $(BUILD) $(LIB)
$(BUILD):
	mkdir -p $(BUILD)
$(LIB):
	mkdir -p $(LIB)

$(BUILD)/%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -I $(INC) -c -o $@ $<

$(OUT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OPENCL_LIB) $(OPENCL_INC) $(SHARED) -o $@ $^

.PHONY: clean
clean:
	rm -f $(BUILD)/*.o $(OUT)
