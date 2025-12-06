############################################
#  Vitis FPGA Build System 
############################################

# Platform (U280)
PLATFORM = xilinx_u280_gen3x16_xdma_1_202211_1

# Kernel name (top function name)
KERNEL_NAME = dct_rgb

# Build target: sw_emu / hw_emu / hw
TARGET ?= hw_emu

# Version tag (v0, v1, v2, ...)
VERSION ?= v0

# Paths
XILINX_VITIS = /share/Xilinx/Vitis/2023.2
XILINX_XRT   = /opt/xilinx/xrt

XRT_INC = $(XILINX_XRT)/include
XRT_LIB = $(XILINX_XRT)/lib

# HLS kernel file (versioned)
HLS_SRC = hls/$(VERSION)_$(KERNEL_NAME)_kernel.cpp

# Output artifacts include version tag
XO_FILE      = build/$(VERSION)_$(KERNEL_NAME)_$(TARGET).xo
XCLBIN_FILE  = build/$(VERSION)_$(KERNEL_NAME)_$(TARGET).xclbin
HOST_EXE     = build/host.exe

HOST_SRC = host/host.cpp

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
# Compile host application
############################################
host: build_dir
	g++ $(HOST_SRC) -o $(HOST_EXE) -O2 \
	    -I$(XRT_INC) \
	    -L$(XRT_LIB) \
	    -lxrt_coreutil -lpthread

############################################
# Build all (kernel + host)
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
	rm -rf build .run _x *.log
