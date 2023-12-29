#***************************************************************************
#
#   File Name   [Makefile]
#
#   Synopsis    []
#
#   Description []
#
#   Revision    [0.1; Initial build from NVIDIA common.mk;
#                Xiao-Long Wu, ECE UIUC
#                0.2; Revision due to the project architecture change;
#                Xiao-Long Wu, ECE UIUC]
#   Date        [02/14/2011]
#
#***************************************************************************

# =========================================================================
# Source files
# =========================================================================

# Target executable name
EXECUTABLE := mriSolver
TARGET     := $(EXECUTABLE)
LIBRARY    := lib$(TARGET).a

# Cuda source files (compiled with nvcc)
CUFILES    := main_mri.cu

# C/C++ source files (compiled with g++)
CCFILES    := main.cpp

# =========================================================================
# Environment setting: Library directories
# =========================================================================

# Define this if you want to include it from other Makefiles.
MRISOLVER ?= .

BINDIR ?= $(MRISOLVER)/bin/$(OSLOWER)
OBJDIR ?= $(MRISOLVER)/obj/$(OSLOWER)
LIBDIR ?= $(MRISOLVER)/lib/$(OSLOWER)

# CUDA SDK code sample library path
export CUDA_SDK_DIR  := $(MRISOLVER)/cuda_sdk
export CUDA_SDK_LIB  := -L$(CUDA_SDK_DIR)/lib/$(OSLOWER) -lcutil -lpthread
export CUDA_SDK_OBJ  = $(CUDA_SDK_DIR)/obj/$(OSLOWER)/$(BINSUBDIR)

export BRUTEFORCE_DIR  := $(MRISOLVER)/bruteForce
export BRUTEFORCE_LIB  ?= -L$(BRUTEFORCE_DIR)/lib/$(OSLOWER) -lbruteForce
export BRUTEFORCE_OBJ  = $(BRUTEFORCE_DIR)/obj/$(OSLOWER)/$(BINSUBDIR)

export TOEPLITZ_DIR  := $(MRISOLVER)/toeplitz
export TOEPLITZ_LIB  ?= -L$(TOEPLITZ_DIR)/lib/$(OSLOWER) -ltoeplitz
export TOEPLITZ_OBJ  = $(TOEPLITZ_DIR)/obj/$(OSLOWER)/$(BINSUBDIR)

export KERNELS_DIR  := $(MRISOLVER)/kernels
export KERNELS_LIB  ?= -L$(KERNELS_DIR)/lib/$(OSLOWER) -lkernels
export KERNELS_OBJ  = $(KERNELS_DIR)/obj/$(OSLOWER)/$(BINSUBDIR)

export DEVICEQUERY_DIR := $(MRISOLVER)/deviceQuery
export DEVICEQUERY_LIB ?= -L$(DEVICEQUERY_DIR)/lib/$(OSLOWER) -ldeviceQuery
export DEVICEQUERY_OBJ  = $(DEVICEQUERY_DIR)/obj/$(OSLOWER)/$(BINSUBDIR)

export XCPPLIB_DIR := $(MRISOLVER)/xcpplib
export XCPPLIB_LIB ?= -L$(XCPPLIB_DIR)/lib/$(OSLOWER) -lxcpplib
export XCPPLIB_OBJ  = $(XCPPLIB_DIR)/obj/$(OSLOWER)/$(BINSUBDIR)

INCLUDES ?= -I$(CUDA_SDK_DIR)/inc -I$(BRUTEFORCE_DIR) -I$(TOEPLITZ_DIR) -I$(KERNELS_DIR) -I$(DEVICEQUERY_DIR) -I$(XCPPLIB_DIR)
# Object files used to contribute the MRISOLVER library
PROJ_OBJS = $(CUDA_SDK_OBJ)/*.o \
            $(BRUTEFORCE_OBJ)/*.o \
            $(KERNELS_OBJ)/*.o \
            $(DEVICEQUERY_OBJ)/*.o \
            $(XCPPLIB_OBJ)/*.o

# =========================================================================
# Environment setting: Compilers
# =========================================================================

# Use special math function units
COMMONFLAGS := -use_fast_math

# Enable multi-threading in CPU code if OpenMP is supported.

#export OPENMP := "true"
export OPENMP := "false"

# Enable double-precision support or not. It is supported in GT200 based
# cards, GTX260/275/280/285/295, Telsa C1060, Telsa S1070, Quadro FX5800
# and the new Fermi based cards.
# If double precision is supported in current device, this should be enabled
# in the Makefile. If not, using single-precision computation is faster.
# Note: Using double-precision support on platforms not supporting it
#       can cause execution failure.

#export DOUBLE_PRECISION := "true"
export DOUBLE_PRECISION := "false"

# Enable to compute FLOPs. This can slow down the performance a little.

#export COMPUTE_FLOPS := "true"
export COMPUTE_FLOPS := "false"

# Compiler setup
# GCC/G++ 4.4 doesn't support CUDA
# GCC/G++ 4.3 supports both OpenMP and CUDA
# GCC/G++ 4.1 supports only CUDA

export CXX        ?= g++
export CC         ?= gcc
export LINKER     ?= g++ -fPIC
export AR         ?= ar
export RANLIB     ?= ranlib

# CUDA Toolkit path

export CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

# =========================================================================
# Rules
# =========================================================================

# Default architecture compute capability
# This will be overrided when DOUBLE_PRECISION is "true".
CUDA_ARCH := -arch sm_10

ifeq ($(OPENMP), "true")
  CXXFLAGS += -fopenmp -DENABLE_OPENMP
  # NVCC doesn't understand -fopenmp but at least we can pass the flag.
  NVCCFLAGS += -DENABLE_OPENMP
  OPENMP_LIB := -fopenmp
endif

ifeq ($(DOUBLE_PRECISION), "true")
  CXXFLAGS += -DENABLE_DOUBLE_PRECISION
  NVCCFLAGS += -DENABLE_DOUBLE_PRECISION
  # Only sm_13 or above support double precision hardware.
  CUDA_ARCH := -arch sm_20
endif

# Override the options given above
CUDA_ARCH := -arch sm_10
CUDA_ARCH := -gencode arch=compute_37,code=sm_37

ifeq ($(COMPUTE_FLOPS), "true")
  CXXFLAGS += -DENABLE_COMPUTE_FLOPS
  NVCCFLAGS += -DENABLE_COMPUTE_FLOPS
endif

NVCCFLAGS += $(CUDA_ARCH)

# =========================================================================
# Environment setting: Others
# =========================================================================

.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx

# Add new SM Versions here as devices with new Compute Capability are released
SM_VERSIONS := sm_10 sm_11 sm_12 sm_13 sm_20

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# detect if 32 bit or 64 bit system
HP_64 =	$(shell uname -m | grep 64)

# Compilers
ifeq "$(strip $(HP_64))" ""
   NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc 
else
   NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc 
endif

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include

# architecture flag for cubin build
CUBIN_ARCH_FLAG :=

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain

# Compiler-specific flags
#NVCCFLAGS := 
CXXFLAGS  += $(CXXWARN_FLAGS)
CFLAGS    += $(CWARN_FLAGS)

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(debug),1)
  ifeq ($(profile),1)
    COMMONFLAGS += -g -pg 
  else
    COMMONFLAGS += -g
  endif
	BINSUBDIR   := debug
	LIBSUFFIX   := D
	NVCCFLAGS   += -DDEBUG -DDEBUG_MRISOLVER --define-macro debug
	CXXFLAGS    += -DDEBUG -DDEBUG_MRISOLVER -Ddebug
	CFLAGS      += -DDEBUG -DDEBUG_MRISOLVER -Ddebug
else
	COMMONFLAGS += -O3
	#COMMONFLAGS += -O3 -march=core2 -msse2 -ftree-vectorize -ftree-vectorizer-verbose=5
	BINSUBDIR   := release
	LIBSUFFIX   :=
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

# append optional arch/SM version flags (such as -arch sm_11)
#NVCCFLAGS += $(SMVERSIONFLAGS)

# architecture flag for cubin build
CUBIN_ARCH_FLAG :=

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)
	ifneq ($(DARWIN),)
		OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU $(CUDA_SDK_DIR)/lib/$(OSLOWER)/libGLEW.a
	else
		OPENGLLIB := -lGL -lGLU -lX11 -lXi -lXmu

		ifeq "$(strip $(HP_64))" ""
			OPENGLLIB += -lGLEW -L/usr/X11R6/lib
		else
			OPENGLLIB += -lGLEW_x86_64 -L/usr/X11R6/lib64
		endif
	endif

	CUBIN_ARCH_FLAG := -m64
endif

ifeq ($(USEGLUT),1)
	ifneq ($(DARWIN),)
		OPENGLLIB += -framework GLUT
	else
		OPENGLLIB += -lglut
	endif
endif

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl$(LIBSUFFIX)
endif

ifeq ($(USERENDERCHECKGL),1)
	RENDERCHECKGLLIB := -lrendercheckgl$(LIBSUFFIX)
endif

ifeq ($(USECUDPP), 1)
	ifeq "$(strip $(HP_64))" ""
		CUDPPLIB := -lcudpp
	else
		CUDPPLIB := -lcudpp64
	endif

	CUDPPLIB := $(CUDPPLIB)$(LIBSUFFIX)

	ifeq ($(emu), 1)
		CUDPPLIB := $(CUDPPLIB)_emu
	endif
endif

# Libs
ifeq "$(strip $(HP_64))" ""
   LIB := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_DIR)/lib/$(OSLOWER) $(OPENMP_LIB) $(CUDA_SDK_LIB) $(BRUTEFORCE_LIB) $(TOEPLITZ_LIB) $(KERNELS_LIB) $(DEVICEQUERY_LIB) $(XCPPLIB_LIB)
else
   LIB := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_DIR)/lib/$(OSLOWER) $(OPENMP_LIB) $(CUDA_SDK_LIB) $(BRUTEFORCE_LIB) $(TOEPLITZ_LIB) $(KERNELS_LIB) $(DEVICEQUERY_LIB) $(XCPPLIB_LIB)
endif

# If dynamically linking to CUDA and CUDART, we exclude the libraries from the LIB
ifeq ($(USECUDADYNLIB),1)
     LIB += ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) -ldl -rdynamic 
else
# static linking, we will statically link against CUDA and CUDART
  ifeq ($(USEDRVAPI),1)
     LIB += -lcuda   ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB)
  else
     LIB += -lcudart ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB)
  endif
endif

# Using CUFFT library
ifeq ($(emu),1)
  LIB += -lcufftemu
else
  LIB += -lcufft
endif

ifeq ($(USECUBLAS),1)
  ifeq ($(emu),1)
    LIB += -lcublasemu
  else
    LIB += -lcublas
  endif
endif

# Device emulation configuration
# FIXME: This is no longer supported.
ifeq ($(emu), 1)
	NVCCFLAGS   += -deviceemu --define-macro emu
	CUDACCFLAGS += 
	BINSUBDIR   := emu$(BINSUBDIR)
	# consistency, makes developing easier
	CXXFLAGS    += -D__DEVICE_EMULATION__ -Demu
	CFLAGS      += -D__DEVICE_EMULATION__ -Demu
endif

TARGETDIR := $(BINDIR)/$(BINSUBDIR)
TARGET    := $(TARGETDIR)/$(TARGET)
LINKLINE   = $(LINKER) -o $(TARGET) $(OBJS) $(LIB) $(COMMONFLAGS)

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

# =========================================================================
# Check for input flags and set compiler flags appropriately
# =========================================================================

ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
	NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif

# =========================================================================
# Set up object files
# =========================================================================

OBJDIR := $(OBJDIR)/$(BINSUBDIR)
OBJS += $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
OBJS += $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CFILES)))
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(CUFILES)))

# =========================================================================
# Set up cubin output files
# =========================================================================

CUBINDIR := $(MRISOLVER)/data
CUBINS += $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

# =========================================================================
# Set up PTX output files
# =========================================================================

PTXDIR := $(MRISOLVER)/data
PTXBINS += $(patsubst %.cu,$(PTXDIR)/%.ptx,$(notdir $(PTXFILES)))

# =========================================================================
# Make Rules
# =========================================================================

$(OBJDIR)/%.c.o : $(MRISOLVER)/%.c $(C_DEPS)
	@echo
	@echo Making $< at $(BINSUBDIR) mode ...
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o : $(MRISOLVER)/%.cpp $(C_DEPS)
	@echo
	@echo Making $< at $(BINSUBDIR) mode ...
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.cu.o : $(MRISOLVER)/%.cu $(CU_DEPS)
	@echo
	@echo Making $< at $(BINSUBDIR) mode ...
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<

$(CUBINDIR)/%.cubin : $(MRISOLVER)/%.cu cubindirectory
	@echo
	@echo Making $< at $(BINSUBDIR) mode ...
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -cubin $<

$(PTXDIR)/%.ptx : $(MRISOLVER)/%.cu ptxdirectory
	@echo
	@echo Making $< at $(BINSUBDIR) mode ...
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -ptx $<

# The following definition is a template that gets instantiated for each SM
# version (sm_10, sm_13, etc.) stored in SMVERSIONS.  It does 2 things:
# 1. It adds to OBJS a .cu_sm_XX.o for each .cu file it finds in CUFILES_sm_XX.
# 2. It generates a rule for building .cu_sm_XX.o files from the corresponding 
#    .cu file.
#
# The intended use for this is to allow Makefiles that use common.mk to compile
# files to different Compute Capability targets (aka SM arch version).  To do
# so, in the Makefile, list files for each SM arch separately, like so:
#
# CUFILES_sm_10 := mycudakernel_sm10.cu app.cu
# CUFILES_sm_12 := anothercudakernel_sm12.cu
#
define SMVERSION_template
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_$(1))))
$(OBJDIR)/%.cu_$(1).o : $(MRISOLVER)/%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $$@ -c $$< $(NVCCFLAGS) -arch $(1)
endef

# This line invokes the above template for each arch version stored in
# SM_VERSIONS.  The call funtion invokes the template, and the eval
# function interprets it as make commands.
$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))

$(TARGET): makedirectories make_cuda_sdk make_bruteForce make_toeplitz make_kernels make_deviceQuery make_xcpplib library $(OBJS) $(CUBINS) $(PTXBINS) Makefile
	$(VERBOSE)@echo
	@echo Making $(TARGET) at $(BINSUBDIR) mode ...
	$(VERBOSE)$(LINKLINE)
	$(VERBOSE)cp $(TARGET) .
	@echo All mriSolver files are compiled.

library: $(OBJS)
	$(VERBOSE)@ echo
	@ echo Making '"'$(LIBDIR)/$(LIBRARY)'"' for $(OSLOWER) platform at $(BINSUBDIR) mode ...
	$(VERBOSE)$(AR) r $(LIBDIR)/$(LIBRARY) $(OBJS) $(PROJ_OBJS)
	$(VERBOSE)$(RANLIB) $(LIBDIR)/$(LIBRARY)

make_cuda_sdk:
	$(VERBOSE)@echo
	@echo Making '"'NVIDIA SDK code sample library'"' at $(BINSUBDIR) mode ...
	$(VERBOSE)$(MAKE) --directory=$(CUDA_SDK_DIR)

make_bruteForce:
	$(VERBOSE)@echo
	@echo Making '"'bruteForce library'"' at $(BINSUBDIR) mode ...
	$(VERBOSE)$(MAKE) --directory=$(BRUTEFORCE_DIR)

make_toeplitz:
	$(VERBOSE)@echo
	@echo Making '"'Toeplitz library'"' at $(BINSUBDIR) mode ...
	$(VERBOSE)$(MAKE) --directory=$(TOEPLITZ_DIR)

make_kernels:
	$(VERBOSE)@echo
	@echo Making '"'kernels library'"' at $(BINSUBDIR) mode ...
	$(VERBOSE)$(MAKE) --directory=$(KERNELS_DIR)

make_deviceQuery:
	$(VERBOSE)@echo
	@echo Making '"'deviceQuery library'"' at $(BINSUBDIR) mode ...
	$(VERBOSE)$(MAKE) --directory=$(DEVICEQUERY_DIR)

make_xcpplib:
	$(VERBOSE)@echo
	@echo Making '"'XCPPLIB library'"' at $(BINSUBDIR) mode ...
	$(VERBOSE)$(MAKE) --directory=$(XCPPLIB_DIR)

cubindirectory:
	$(VERBOSE)mkdir -p $(CUBINDIR)

ptxdirectory:
	$(VERBOSE)mkdir -p $(PTXDIR)

makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)

tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(EXECUTABLE)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(LIBDIR)/$(LIBRARY)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(PTXBINS)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)

clean_cuda_sdk :
	$(VERBOSE)$(MAKE) clean --directory=$(CUDA_SDK_DIR)
clean_bruteForce :
	$(VERBOSE)$(MAKE) clean --directory=$(BRUTEFORCE_DIR)
clean_toeplitz :
	$(VERBOSE)$(MAKE) clean --directory=$(TOEPLITZ_DIR)
clean_kernels :
	$(VERBOSE)$(MAKE) clean --directory=$(KERNELS_DIR)
clean_deviceQuery :
	$(VERBOSE)$(MAKE) clean --directory=$(DEVICEQUERY_DIR)
clean_xcpplib :
	$(VERBOSE)$(MAKE) clean --directory=$(XCPPLIB_DIR)

clean_all : clean_cuda_sdk clean_bruteForce clean_toeplitz clean_kernels clean_deviceQuery clean_xcpplib tidy clean

clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)

# =========================================================================
#  End of Makefile
# =========================================================================

