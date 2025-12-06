############################################
#  Vitis FPGA Build System 
############################################

PLATFORM = xilinx_u280_gen3x16_xdma_1_202211_1
KERNEL_NAME = dct_rgb
TARGET ?= hw_emu

XILINX_VITIS = /share/Xilinx/Vitis/2023.2
XILINX_XRT   = /opt/xilinx/xrt

XRT_INC = $(XILINX_XRT)/include
XRT_LIB = $(XILINX_XRT)/lib

# ---- FIXED: No versioning unless files exist ----
HLS_SRC = hls/dct_rgb_kernel.cpp

XO_FILE      = build/$(KERNEL_NAME)_$(TARGET).xo
XCLBIN_FILE  = build/$(KERNEL_NAME)_$(TARGET).xclbin

HOST_SRC     = host/host.cpp
HOST_EXE     = build/host.exe

############################################
build_dir:
	mkdir -p build

############################################
xo: build_dir
	v++ -c -t $(TARGET) \
	    --platform $(PLATFORM) \
	    -k $(KERNEL_NAME) \
	    -o $(XO_FILE) \
	    $(HLS_SRC)

############################################
xclbin: xo
	v++ -l -t $(TARGET) \
	    --platform $(PLATFORM) \
	    $(XO_FILE) \
	    -o $(XCLBIN_FILE)

############################################
host: build_dir
	g++ $(HOST_SRC) -o $(HOST_EXE) -O2 \
	    -I$(XRT_INC) \
	    -L$(XRT_LIB) \
	    -lxrt_coreutil -lpthread

############################################
all: xo xclbin host

############################################
sw_emu:
	make TARGET=sw_emu all

hw_emu:
	make TARGET=hw_emu all

hw:
	make TARGET=hw all

############################################
clean:
	rm -rf build *.log _x .run
