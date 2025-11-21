############################################
#  Vitis FPGA Build System
############################################

PLATFORM = xilinx_u280_gen3x16_xdma_1_202211_1
KERNEL_NAME = dct

# Targets: sw_emu / hw_emu / hw
TARGET ?= hw_emu

HLS_SRC      = hls/$(KERNEL_NAME).cpp
XO_FILE      = build/$(KERNEL_NAME)_$(TARGET).xo
XCLBIN_FILE  = build/$(KERNEL_NAME)_$(TARGET).xclbin

HOST_SRC     = host/host.cpp
HOST_EXE     = build/host.exe

# XRT dirs
XRT_INC = $(XILINX_XRT)/include
XRT_LIB = $(XILINX_XRT)/lib

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
