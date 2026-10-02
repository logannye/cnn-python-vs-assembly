# All numerical and runtime objects are handwritten ARM64 assembly.
CLANG := xcrun clang
BUILD := build
FLAGS := -arch arm64 -g
SOURCES := convolution.S model.S
OBJECTS := $(SOURCES:%.S=$(BUILD)/%.o)

.PHONY: all clean
all: $(BUILD)/cnn-assembly $(BUILD)/libcnn.dylib

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: %.S | $(BUILD)
	$(CLANG) $(FLAGS) -c $< -o $@

$(BUILD)/cnn-assembly: $(OBJECTS) $(BUILD)/runtime.o Makefile
	$(CLANG) -arch arm64 $(OBJECTS) $(BUILD)/runtime.o -o $@

$(BUILD)/libcnn.dylib: $(OBJECTS) Makefile
	$(CLANG) -arch arm64 -dynamiclib $(OBJECTS) -o $@

clean:
	rm -f $(BUILD)/*.o $(BUILD)/cnn-assembly $(BUILD)/libcnn.dylib
