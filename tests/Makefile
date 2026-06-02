CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -fPIC -fopenmp -ffp-contract=off \
	-ffloat-store -mavx512f -mfma \
	-fno-builtin-exp -fno-builtin-log -fno-builtin-sin \
	-fno-builtin-cos -fno-builtin-tan -fno-builtin-pow \
	-fno-builtin-sqrt -fno-builtin-atan2 -fno-builtin-log2 \
	-fno-builtin-log10 -fno-builtin-asin -fno-builtin-acos \
	-fno-builtin-atan -fno-builtin-exp2

INCLUDES  = -I.. -I../pycpp -I../scipy \
            -I/usr/include/eigen3 \
            -I/usr/include/numpycpp \
            -I/usr/include/pocketfft \
            $(shell python3 -m pybind11 --includes)
LDFLAGS   = -shared -ldl

MODULE    = scipycpp.so
SRC       = module.cpp

.PHONY: all clean test

all: $(MODULE)

$(MODULE): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

test: $(MODULE)
	@cd .. && PYTHONPATH=tests:$$PYTHONPATH python3 -m pytest tests/test_all.py -q --tb=short --no-header

clean:
	rm -f $(MODULE)
