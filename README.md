# Symmetry-Adapted DMRG Library

A unified C++17 template library for Density Matrix Renormalization Group (DMRG)
calculations with U(1) and SU(2) symmetry, supporting both real (`double`) and
complex (`dcmplex`) wavefunctions.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Class Architecture](#class-architecture)
4. [Symmetry Policy Pattern](#symmetry-policy-pattern)
5. [Template Instantiation Strategy](#template-instantiation-strategy)
6. [Global Variables](#global-variables)
7. [Building](#building)
8. [Running the Programs](#running-the-programs)
9. [Algorithm Flow](#algorithm-flow)

---

## Overview

This library consolidates what were previously eight separate codebases
(two scalars × two symmetries × two levels of structure) into a single
composable template design. The four executable targets are:

| Executable | Scalar | Symmetry | Use case |
|---|---|---|---|
| `dmrg_su2_real` | `double` | SU(2) | Time-reversal-invariant Hamiltonians |
| `dmrg_su2_complex` | `dcmplex` | SU(2) | General spin models with complex hopping |
| `dmrg_u1_real` | `double` | U(1) | Real wavefunctions, conserved Sz |
| `dmrg_u1_complex` | `dcmplex` | U(1) | Complex wavefunctions, conserved Sz |

All four targets share the same compiled object files (`COMMON_OBJS`).
Only the main program and link-time macro (`-DUSE_SU2` or `-DUSE_U1`) differ.

---

## Directory Structure

```
mps/
├── makefile
│
├── dense.hpp                    # Dense<Scalar>: rank-N tensor (real or complex)
├── block_tensor.hpp             # BlockTensor<Scalar,Sym>: block-sparse symmetry tensor
├── dmrg.hpp                     # DMRG<Scalar,Sym> and Lanczos<Scalar,Sym> declarations
├── lanczos.hpp                  # Lanczos eigensolver variants
├── cgc_tables.hpp               # CGCTables struct and builder declaration
├── scalar.hpp                   # BLAS/LAPACK dispatch traits
│
├── symmetry/
│   ├── su2.hpp                  # SU2Bond, SU2Struct, CGCTable, SU2Symmetry policy
│   └── u1.hpp                   # U1Bond, U1Struct, U1Symmetry policy
│
├── src/
│   ├── dense_inst.cpp           # Dense<double> and Dense<dcmplex> instantiations
│   ├── block_tensor_inst.cpp    # BlockTensor<Scalar,Sym> full implementation (~75 KB)
│   ├── dmrg_inst.cpp            # DMRG<Scalar,Sym>, Lanczos<Scalar,Sym> implementation
│   ├── cgc_tables.cpp           # makeup_clebsch_gordan_coefficient_tensors()
│   ├── cgc_tables_compare.cpp   # validation utility for CGC tables
│   └── symmetry/
│       ├── su2.cpp              # SU2Bond, SU2Struct, CGCTable implementations
│       └── u1.cpp               # U1Bond, U1Struct implementations
│
├── main/
│   ├── main_dmrg_su2_real.cpp
│   ├── main_dmrg_su2_complex.cpp
│   ├── main_dmrg_u1_real.cpp
│   └── main_dmrg_u1_complex.cpp
│
└── two_layer/                   # Two-layer MPS ansatz (experimental)
    ├── bottom_mps.hpp
    ├── top_mps.hpp
    └── two_layer_dmrg.hpp
```

---

## Class Architecture

### Layer 1 — Dense tensor (`dense.hpp`, `src/dense_inst.cpp`)

```cpp
template<typename Scalar>   // double or dcmplex
class Dense {
    int     nbond;           // tensor rank
    int     nelem;           // total number of elements
    int    *bonddim;         // dimension of each index
    Scalar *data;            // C-order flattened array

    // Core numerical operations
    void svd(Dense& U, Dense& V, double* svals, ...);
    Dense& contract(const Dense& A, int ia, const Dense& B, int ib);
    void hermitian_eig(double* evals);

    // DMRG environment contractions
    Dense& contract_dmrg_overlap_initial(...);
    Dense& contract_dmrg_operator_initial(...);
    // ... (8 total environment update methods)

    // SU(2)-specific helpers
    void make_cgc(int j1, int j2, int j3);
    void multiply_cgc(int j1, int j2, int j3, int bond, int dir);
};

extern template class Dense<double>;
extern template class Dense<dcmplex>;
```

`Dense` is the numerical workhorse. All block-level arithmetic in `BlockTensor`
delegates to `Dense` operations.

---

### Layer 2 — Symmetry descriptors (`symmetry/su2.hpp`, `symmetry/u1.hpp`)

Each symmetry provides two descriptor classes — one for a single bond and one
for an entire tensor — plus a static policy class.

#### SU(2)

```
SU2Bond
  bonddir       — +1 (outgoing) or -1 (incoming)
  nmoment       — number of J sectors on this bond
  angularmoment — 2J values, e.g. [0, 1, 2] for J=0,½,1
  bonddim       — reduced dimension for each J sector
  cgcdim        — 2J+1 (CGC multiplicity = number of m_J states)

SU2Struct  (metadata for a full rank-3 MPS tensor)
  nbond         — number of bonds (3 for standard site tensor)
  nten          — number of symmetry-allowed blocks
  locspin       — 2J of the physical index
  bonddir[i]    — direction of bond i
  angularmoment[i][j], bonddim[i][j], cgcdim[i][j]
```

The Wigner–Eckart theorem is applied: the full tensor element decomposes as

```
A^J_{m_L m_P m_R}  =  Â^J · <J_L m_L; J_P m_P | J_R m_R>
```

where `Â^J` is the reduced matrix element (stored in `data_blocks`) and
the Clebsch–Gordan coefficient `<…|…>` is stored separately in `cgc_blocks`.

#### U(1)

```
U1Bond
  bonddir       — +1 or -1
  nmoment       — number of charge sectors
  angularmoment — charge values (integer, e.g. 2Sz)
  bonddim       — dimension of each sector

U1Struct
  nbond, nten, locspin
  bonddir[i], angularmoment[i][j], bonddim[i][j]
  (no cgcdim — CGC multiplicities are trivially 1)
```

---

### Layer 3 — Block-sparse tensor (`block_tensor.hpp`, `src/block_tensor_inst.cpp`)

```cpp
template<typename Scalar, typename Sym>
class BlockTensor {
    using BondType   = typename Sym::BondType;    // SU2Bond or U1Bond
    using StructType = typename Sym::StructType;  // SU2Struct or U1Struct

    int nbond, nten, locspin;
    Dense<Scalar> *data_blocks;  // [nten]  reduced matrix elements (SU2) or full (U1)
    Dense<double> *cgc_blocks;   // [nten]  CGC tensors; only allocated when Sym::has_cgc
    StructType     cgc;          // symmetry metadata

    // DMRG environment updates
    void operator_initial(...);
    void operator_transformation(...);
    void overlap_initial(...);
    void overlap_transformation(...);
    void hamiltonian_vector_multiplication(...);

    // Linear algebra
    void svd(BlockTensor& U, double cut, BlockTensor& V, double cut2, double* svals);
    void contract(const BlockTensor& A, int ia, const BlockTensor& B, int ib);
    void fuse(const BondType& a, const BondType& b);

    // Canonical form transformations
    BlockTensor& left2right_vectran();
    BlockTensor& right2left_vectran();
};
```

All internal branches between SU(2) and U(1) are resolved at compile time via
`if constexpr (Sym::has_cgc)`.

---

### Layer 4 — DMRG solver (`dmrg.hpp`, `src/dmrg_inst.cpp`)

```cpp
template<typename Scalar, typename Sym>
class DMRG {
    using TensorType  = BlockTensor<Scalar, Sym>;
    using LanczosType = Lanczos<Scalar, Sym>;

    int ly, lx, ns;       // lattice geometry (ns = lx * ly total sites)
    int bondd;            // max bond dimension
    int totspin, exci;    // target quantum sector and excitation index

    TensorType *uu;       // [ns]  MPS site tensors
    TensorType *hh;       // [ns]  effective Hamiltonian blocks
    TensorType **opr;     // [ns][nbb]  operator environment blocks
    TensorType **ovlp;    // [ns][max_exci]  overlap environments
    TensorType **orth;    // [ns][max_exci]  stored excited states
    TensorType  sigma[4]; // local spin operators
    double    **ww;       // [max_exci][ns]  singular value spectra
    double      gs_enr[100];

    void do_idmrg();
    void prepare_sweep();
    void sweep();
    bool read_mps(int dr, int dge);
    void save_mps2(int exci);
    double get_enr() const;
};
```

---

### CGC Tables (`cgc_tables.hpp`, `src/cgc_tables.cpp`)

Relevant only for SU(2). The CGC factor tables are precomputed once and stored
globally to avoid recomputation across sweeps.

```cpp
struct CGCTables {
    int physpn, max_angm;

    double ***fac_hamilt_vec;                   // [M][M][M]
    double *****fac_operator_onsite_left/rght;
    double ******fac_operator_transformation_left/rght;
    double *****fac_operator_pairup_left/rght;
    double *****fac_permutation_left/rght;
};

// Entry point — called once in DMRG constructor for SU(2):
void makeup_clebsch_gordan_coefficient_tensors(int physpn, int max_angm, CGCTables& t);
```

---

## Symmetry Policy Pattern

The `Sym` template parameter is a **compile-time policy class** with a static
interface:

```cpp
struct SU2Symmetry {
    using BondType   = SU2Bond;
    using StructType = SU2Struct;
    static constexpr bool has_cgc = true;   // allocate cgc_blocks, use CGC factors
    static bool triangle(int j1, int j2, int j3);
    static bool block_allowed(int j1, int d1, int j2, int d2, int j3, int d3);
};

struct U1Symmetry {
    using BondType   = U1Bond;
    using StructType = U1Struct;
    static constexpr bool has_cgc = false;  // no extra CGC storage
    static bool charge_allowed(int q0, int d0, int q1, int d1, int q2, int d2);
};
```

The `has_cgc` flag drives the key compile-time branches:

```cpp
// Memory allocation in BlockTensor::set()
if constexpr (Sym::has_cgc)
    cgc_blocks = new Dense<double>[nten];

// Contraction in hamiltonian_vector_multiplication()
if constexpr (Sym::has_cgc) {
    // SU(2): contract data_blocks with CGC factors from g_cgc_tables
} else {
    // U(1): direct block-diagonal matrix-vector product
}

// Helper for tensor argument extraction
template<typename Sym>
static bool get_ta(const typename Sym::StructType& s, int i,
                   int* angm, int* bdim, int* cdim) {
    if constexpr (Sym::has_cgc)
        return s.get_tensor_argument(i, angm, bdim, cdim); // 4-arg: SU(2) with cdim
    else {
        bool r = s.get_tensor_argument(i, angm, bdim);     // 3-arg: U(1), cdim = 1
        for (int k = 0; k < s.get_nbond(); ++k) cdim[k] = 1;
        return r;
    }
}
```

The `BondType` and `StructType` aliases ensure that `BlockTensor<Scalar, Sym>`
always uses the correct quantum-number classes without any virtual dispatch.

---

## Template Instantiation Strategy

Templates are defined in header files and **explicitly instantiated** in the
corresponding `.cpp` files. This keeps compile times manageable and gives one
well-defined location for each specialization.

### `src/block_tensor_inst.cpp`

```cpp
// At the end of the file:
template class BlockTensor<double,  SU2Symmetry>;
template class BlockTensor<dcmplex, SU2Symmetry>;
template class BlockTensor<double,  U1Symmetry>;
template class BlockTensor<dcmplex, U1Symmetry>;
```

### `src/dmrg_inst.cpp`

```cpp
template class DMRG<double,  SU2Symmetry>;
template class DMRG<dcmplex, SU2Symmetry>;
template class DMRG<double,  U1Symmetry>;
template class DMRG<dcmplex, U1Symmetry>;

template class Lanczos<double,  SU2Symmetry>;
template class Lanczos<dcmplex, SU2Symmetry>;
template class Lanczos<double,  U1Symmetry>;
template class Lanczos<dcmplex, U1Symmetry>;
```

The header files declare these as `extern template` to prevent redundant
instantiation in every translation unit that includes them:

```cpp
// block_tensor.hpp
extern template class BlockTensor<double,  SU2Symmetry>;
extern template class BlockTensor<dcmplex, SU2Symmetry>;
// ... etc.
```

---

## Global Variables

Each main program defines the following globals, which are declared `extern`
in the implementation files:

```cpp
int        max_dcut     = 64;   // maximum bond dimension; overridden by -d flag
int        psize        = 1;    // OpenMP thread count; overridden by -psize flag
int        myrank       = 0;    // MPI rank (always 0 in single-process mode)
CGCTables* g_cgc_tables = nullptr; // CGC factor tables; allocated inside DMRG constructor (SU2 only)
```

`myrank` is reserved for future MPI support; in single-process mode it is always 0.

---

## Building

### Requirements

- g++ 14 (or compatible C++17 compiler)
- BLAS and LAPACK (`-lblas -llapack`)
- OpenMP

### Targets

```bash
# Build all four executables
make all

# Build individually
make dmrg_su2_real
make dmrg_su2_complex
make dmrg_u1_real
make dmrg_u1_complex

# Clean all build artifacts
make clean
```

### Compiler flags

| Flag | Effect |
|---|---|
| `-O3 -std=c++17` | Optimized C++17 compilation |
| `-fopenmp` | Enable OpenMP threading |
| `-DUSE_SU2` | Link-time flag for SU(2) targets |
| `-DUSE_U1` | Link-time flag for U(1) targets |

---

## Running the Programs

All four executables share the same command-line interface.

### Command-line arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `-lx` | int | 4 | Chain length (x-direction) |
| `-ly` | int | 4 | Chain width (y-direction; use `ly=1` for 1D) |
| `-d` | int | 64 | Maximum bond dimension D |
| `-sec` | int | 0 | Target quantum sector (see below) |
| `-exci` | int | 0 | Excitation index (0 = ground state) |
| `-niter` | int | 2 | Number of finite DMRG sweeps |
| `-psize` | int | 1 | Number of OpenMP threads |
| `-dr` | int | 0 | Bond dimension of an existing MPS to load (0 = start fresh) |
| `-dge` | int | 0 | Excitation index of existing state to load |

### The `-sec` argument

**SU(2) targets** (`dmrg_su2_real`, `dmrg_su2_complex`):
`-sec` is the total spin **2J** (must be a non-negative integer):

| `-sec` | Sector | Description |
|---|---|---|
| `0` | J=0 | Singlet |
| `1` | J=½ | Doublet |
| `2` | J=1 | Triplet |

**U(1) targets** (`dmrg_u1_real`, `dmrg_u1_complex`):
`-sec` is the total **2Sz**:

| `-sec` | Sector |
|---|---|
| `0` | Sz=0 |
| `2` | Sz=+1 |
| `-2` | Sz=-1 |

### Examples

```bash
# SU(2) real: 1D Heisenberg chain, L=8, D=64, singlet sector, 10 sweeps
./dmrg_su2_real -lx 8 -ly 1 -d 64 -sec 0 -niter 10

# SU(2) complex: same, with 4 OpenMP threads
./dmrg_su2_complex -lx 8 -ly 1 -d 64 -sec 0 -niter 10 -psize 4

# U(1) real: 1D Heisenberg chain, Sz=0 sector, D=128
./dmrg_u1_real -lx 8 -ly 1 -d 128 -sec 0 -niter 20

# U(1) complex: 2D 4×4 lattice, D=200, 4 threads
./dmrg_u1_complex -lx 4 -ly 4 -d 200 -sec 0 -niter 10 -psize 4

# Read a previously saved MPS (D=64) and continue sweeping
./dmrg_su2_real -lx 8 -ly 1 -d 128 -sec 0 -niter 5 -dr 64

# Compute the first excited state (exci=1)
./dmrg_su2_real -lx 8 -ly 1 -d 64 -sec 0 -exci 1 -niter 10
```

---

## Algorithm Flow

### Infinite DMRG (`do_idmrg`)

Grows the chain from 2 sites to `ns = lx * ly` sites:

```
for site i = 1 to ns/2 - 1:
  1. Lanczos diagonalization of the two-site effective Hamiltonian
       → two-site eigenvector (ground or excited state)

  2. SVD: split two-site tensor into uu[i] and uu[ns-1-i]
       → singular values stored in ww[exci][i]
       → bond dimension truncated to max_dcut

  3. wavefunc_transformation: move the canonical centre
       → uu[0..i] become left-orthogonal
       → uu[i+1] is the active site

  4. Update environment blocks (operator and overlap)
       → opr[i+1] contracted from opr[i], uu[i], MPO
       → ovlp[i+1] contracted from ovlp[i], uu[i], uu_ref[i]
```

### Finite DMRG (`prepare_sweep` + `sweep`)

Called after `do_idmrg` to refine the MPS:

```
prepare_sweep():
  Build all left and right environment blocks from scratch.
  Set canonical centre.

sweep():
  Forward pass (left → right):
    for each bond (site_l, site_l+1):
      a. Lanczos diagonalization (two-site)
      b. SVD and truncate → update uu[site_l], uu[site_l+1]
      c. wavefunc_transformation (move right)
      d. Update left environment blocks

  Backward pass (right → left):
    Same, moving from right to left.
```

Convergence is monitored by comparing the energy between consecutive sweeps.
The main programs exit when `|E_new - E_old| < 1e-6`.
