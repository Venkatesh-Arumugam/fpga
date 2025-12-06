############################################
#  Vitis FPGA Build System (Corrected)
############################################

# Platform (U280)
PLATFORM = xilinx_u280_gen3x16_xdma_1_202211_1

# Kernel name
KERNEL_NAME = dct_rgb

# Build target: sw_emu / hw_emu / hw
TARGET ?= hw_emu

# Paths
XILINX_VITIS = /share/Xilinx/Vitis/2023.2
XILINX_XRT   = /opt/xilinx/xrt

XRT_INC = $(XILINX_XRT)/include
XRT_LIB = $(XILINX_XRT)/lib

# Input sources
HLS_SRC = hls/$(VERSION)_$(KERNEL_NAME)_kernel.cpp
XO_FILE      = build/$(KERNEL_NAME)_$(TARGET).xo
XCLBIN_FILE  = build/$(KERNEL_NAME)_$(TARGET).xclbin

HOST_SRC     = host/host.cpp
HOST_EXE     = build/host.exe

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
# Build everything (kernel + host)
############################################
all: xo xclbin host

############################################
# Convenience targets
############################################
sw_emu:
	make TARGET=sw_emu all

hw_emu:
	make TARGET=hw_emu all

hw:
	make TARGET=hw all

############################################
# Clean
############################################
clean:
	rm -rf build *.log _x .run
