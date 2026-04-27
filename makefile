# makefile — unified MPS/DMRG library
#
# Targets
# -------
#   make dmrg_su2_complex    — complex SU2 DMRG
#   make dmrg_su2_real       — real SU2 DMRG
#   make dmrg_u1_complex     — complex U1 DMRG
#   make dmrg_u1_real        — real U1 DMRG
#   make all                 — build all four
#
# Compiler flags
# --------------
#   -DUSE_MPI    : enable MPI (requires mpi.h and MPI-linked BLAS)
#   -DUSE_OMP    : enable OpenMP (default on)
#   -DUSE_SU2    : compile BlockTensor<Scalar, SU2Symmetry> variants
#   -DUSE_U1     : compile BlockTensor<Scalar, U1Symmetry>  variants

CXX      = g++-14
CXXFLAGS = -O3 -std=c++17 -fopenmp
LDFLAGS  = -lblas -llapack -lstdc++

# BLAS/LAPACK on macOS (Accelerate):
# LDFLAGS = -framework Accelerate

INCLUDES = -I.

# ── Common object files ───────────────────────────────────────────────────────
COMMON_OBJS = \
    src/symmetry/u1.o     \
    src/symmetry/su2.o    \
    src/dense_inst.o      \
    src/cgc_tables.o      \
    src/block_tensor_inst.o \
    src/dmrg_inst.o

# ── Per-target main files ─────────────────────────────────────────────────────
MAIN_SU2_COMPLEX = main/main_dmrg_su2_complex.o
MAIN_SU2_REAL    = main/main_dmrg_su2_real.o
MAIN_U1_COMPLEX  = main/main_dmrg_u1_complex.o
MAIN_U1_REAL     = main/main_dmrg_u1_real.o

# ── Build rules ───────────────────────────────────────────────────────────────
all: dmrg_su2_complex dmrg_su2_real dmrg_u1_complex dmrg_u1_real

dmrg_su2_complex: $(COMMON_OBJS) $(MAIN_SU2_COMPLEX)
	$(CXX) $(CXXFLAGS) -DUSE_SU2 -o $@ $^ $(LDFLAGS)

dmrg_su2_real: $(COMMON_OBJS) $(MAIN_SU2_REAL)
	$(CXX) $(CXXFLAGS) -DUSE_SU2 -o $@ $^ $(LDFLAGS)

dmrg_u1_complex: $(COMMON_OBJS) $(MAIN_U1_COMPLEX)
	$(CXX) $(CXXFLAGS) -DUSE_U1 -o $@ $^ $(LDFLAGS)

dmrg_u1_real: $(COMMON_OBJS) $(MAIN_U1_REAL)
	$(CXX) $(CXXFLAGS) -DUSE_U1 -o $@ $^ $(LDFLAGS)

# MPI variant (add -DUSE_MPI and link MPI)
dmrg_su2_complex_mpi: CXX = mpicxx
dmrg_su2_complex_mpi: CXXFLAGS += -DUSE_MPI
dmrg_su2_complex_mpi: $(COMMON_OBJS) $(MAIN_SU2_COMPLEX)
	$(CXX) $(CXXFLAGS) -DUSE_SU2 -o $@ $^ $(LDFLAGS)

# ── Pattern rule ──────────────────────────────────────────────────────────────
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(COMMON_OBJS) \
	    $(MAIN_SU2_COMPLEX) $(MAIN_SU2_REAL) $(MAIN_U1_COMPLEX) $(MAIN_U1_REAL) \
	    dmrg_su2_complex dmrg_su2_real dmrg_u1_complex dmrg_u1_real

.PHONY: all clean
