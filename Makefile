############################################
#  Vitis FPGA Build System (Multi-Variant)
############################################

# Platform
PLATFORM = xilinx_u280_gen3x16_xdma_1_202211_1

# Kernel name (same for all variants)
KERNEL_NAME = dct_accel

# Choose variant: v1, v2, v3, v4, v5
# Example: make VERSION=v3 all
VERSION ?= v1

# Targets: sw_emu / hw_emu / hw
TARGET ?= hw_emu

############################################
# Source files
############################################
HLS_SRC      = hls/$(VERSION)_$(KERNEL_NAME).cpp
XO_FILE      = build/$(VERSION)_$(KERNEL_NAME)_$(TARGET).xo
XCLBIN_FILE  = build/$(VERSION)_$(KERNEL_NAME)_$(TARGET).xclbin

HOST_SRC     = host/host.cpp
HOST_EXE     = build/host.exe

############################################
# XRT and include dirs
############################################
XRT_INC = $(XILINX_XRT)/include
XRT_LIB = $(XILINX_XRT)/lib

HOST_INC = -I$(XRT_INC) -Ihost

############################################
# Build directory
############################################
build_dir:
	mkdir -p build

############################################
# Compile HLS → XO
############################################
xo: build_dir
	v++ -c -t $(TARGET) \
	    --platform $(PLATFORM) \
	    -k $(KERNEL_NAME) \
	    -o $(XO_FILE) \
	    $(HLS_SRC)

############################################
# Link XO → XCLBIN
############################################
xclbin: xo
	v++ -l -t $(TARGET) \
	    --platform $(PLATFORM) \
	    $(XO_FILE) \
	    -o $(XCLBIN_FILE)

############################################
# Compile host
############################################
host: build_dir
	g++ $(HOST_SRC) -o $(HOST_EXE) -O2 \
	    $(HOST_INC) \
	    -L$(XRT_LIB) \
	    -lxrt_coreutil -lpthread

############################################
# Build everything
############################################
all: xo xclbin host

############################################
# Convenience targets
############################################
sw_emu:
	make TARGET=sw_emu VERSION=$(VERSION) all

hw_emu:
	make TARGET=hw_emu VERSION=$(VERSION) all

hw:
	make TARGET=hw VERSION=$(VERSION) all

############################################
# Clean
############################################
clean:
	rm -rf build *.log _x .run
