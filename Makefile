############################################
#  Vitis FPGA Build System
############################################

PLATFORM = xilinx_u280_xdma_201920_3

KERNEL_NAME = dct_rgb

HLS_SRC = hls/$(KERNEL_NAME).cpp

XO_FILE = build/$(KERNEL_NAME).xo
XCLBIN_FILE = build/$(KERNEL_NAME).xclbin

HOST_SRC = host/host.cpp
HOST_EXE = build/host.exe

# XRT paths
XRT_INC = $(XILINX_XRT)/include
XRT_LIB = $(XILINX_XRT)/lib

############################################
# Build directory
############################################
build_dir:
	mkdir -p build

############################################
# Build HLS kernel → XO
############################################
xo: build_dir
	v++ -c -t hw_emu \
	    --platform $(PLATFORM) \
	    -k $(KERNEL_NAME) \
	    -o $(XO_FILE) \
	    $(HLS_SRC)

############################################
# Link → XCLBIN
############################################
xclbin: xo
	v++ -l -t hw_emu \
	    --platform $(PLATFORM) \
	    $(XO_FILE) \
	    -o $(XCLBIN_FILE)

############################################
# Build host program
############################################
host: build_dir
	g++ $(HOST_SRC) -o $(HOST_EXE) -O2 \
	    -I$(XRT_INC) \
	    -L$(XRT_LIB) \
	    -lxrt_coreutil -lpthread

############################################
# Build everything
############################################
all: xo xclbin host

############################################
# Clean
############################################
clean:
	rm -rf build
