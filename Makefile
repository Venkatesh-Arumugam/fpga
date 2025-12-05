############################################
#  Vitis FPGA Build System (Updated for DCT RGB Project)
############################################

# FPGA platform on CloudLab/NERC U280 nodes
PLATFORM = xilinx_u280_gen3x16_xdma_1_202211_1

# Kernel name and sources
KERNEL_NAME = dct_rgb
HLS_SRC     = dct_rgb_kernel.cpp

# Host source
HLS_SRC = hls/dct_rgb_kernel.cpp
HOST_SRC = host/host.cpp

# Output files
TARGET ?= hw        # default build target = hardware
XO_FILE     = build/$(KERNEL_NAME)_$(TARGET).xo
XCLBIN_FILE = build/$(KERNEL_NAME)_$(TARGET).xclbin

# XRT include / lib (auto-discovered from environment)
XRT_INC = $(XILINX_XRT)/include
XRT_LIB = $(XILINX_XRT)/lib

############################################
# Build directory
############################################
build_dir:
	mkdir -p build

############################################
# Compile Kernel (HLS → XO)
############################################
xo: build_dir
	v++ -c -t $(TARGET) \
	    --platform $(PLATFORM) \
	    -k $(KERNEL_NAME) \
	    -o $(XO_FILE) \
	    $(HLS_SRC)

############################################
# Link Kernel (XO → XCLBIN)
############################################
xclbin: xo
	v++ -l -t $(TARGET) \
	    --platform $(PLATFORM) \
	    $(XO_FILE) \
	    -o $(XCLBIN_FILE)

############################################
# Build Host Executable
############################################
host: build_dir
	g++ $(HOST_SRC) -o $(HOST_EXE) -O2 \
	    -I$(XRT_INC) \
	    -L$(XRT_LIB) \
	    -lxrt_coreutil -pthread

############################################
# Build Everything
############################################
all: xo xclbin host

############################################
# Convenience Targets
############################################
sw_emu:
	$(MAKE) TARGET=sw_emu all

hw_emu:
	$(MAKE) TARGET=hw_emu all

hw:
	$(MAKE) TARGET=hw all

############################################
# Clean
############################################
clean:
	rm -rf build *.log *.jou _x .run
