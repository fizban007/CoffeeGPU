###########################
SHELL=/bin/sh

FC := h5pfc
PFLAGS := -DMPI -DHDF5
FFLAGS := -free -O3 -ipo -traceback

EXE_DIR := exec/
SRC_DIR := src/
PREC_SRC_DIR := src_/
EXE_FILE := ffree3D

MAIN_FILE := $(SRC_DIR)ffree3D.F90

SRC_FILES := $(MAIN_FILE) \
             $(SRC_DIR)algorithm.f90 \
             $(SRC_DIR)auxiliary.f90 \
						 $(SRC_DIR)output.f90 \
             $(SRC_DIR)boundary_conditions.f90 \
             $(SRC_DIR)user_bc_loop.f90 \
						 $(SRC_DIR)user_bc_uniform.f90 \
             $(SRC_DIR)user_init_loop.f90 \
						 $(SRC_DIR)user_init_uniform.f90

EXECUTABLE := $(EXE_DIR)$(EXE_FILE)

###########################
.PHONY : all clean

all: dirs precomp compile

dirs:
	mkdir -p $(EXE_DIR) $(PREC_SRC_DIR)

precomp:
	$(foreach fl, $(SRC_FILES), cpp -nostdinc -C -P -w $(PFLAGS) $(fl) > $(addprefix $(PREC_SRC_DIR), $(notdir $(fl)));)

compile:
	$(FC) -v $(addprefix $(PREC_SRC_DIR), $(notdir $(MAIN_FILE))) $(FFLAGS) -o $(EXECUTABLE) $(PFLAGS)
	rm *.mod
	rm *.o

clean:
	rm -f *.mod
	rm -rf $(PREC_SRC_DIR)
	rm -rf $(EXE_DIR)
