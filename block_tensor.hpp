/**
 * block_tensor.hpp — Block-sparse symmetry-adapted tensor.
 *
 * Replaces (all four variants with one template):
 *   tensor_su2.hpp / .cpp   (mps_real)  — BlockTensor<double,  SU2Symmetry>
 *   tensor_su2.hpp / .cpp   (mps_u1)    — BlockTensor<dcmplex, SU2Symmetry>
 *   tensor_u1.hpp  / .cpp   (mps_u1)    — BlockTensor<dcmplex, U1Symmetry>
 *   (new)                               — BlockTensor<double,  U1Symmetry>
 *
 * Memory layout
 * -------------
 * A BlockTensor stores nten dense blocks.  For each allowed block b:
 *
 *   SU2:  data_block[b]  ← Dense<Scalar>  of reduced matrix elements (Wigner-Eckart)
 *         cgc_block[b]   ← Dense<double>  of Clebsch-Gordan coefficients (real always)
 *
 *   U1:   data_block[b]  ← Dense<Scalar>  of full matrix elements
 *         (no cgc_block)
 *
 * Template parameters
 * -------------------
 *   Scalar  = double or dcmplex  (real or complex wavefunction)
 *   Sym     = U1Symmetry or SU2Symmetry  (see symmetry/u1.hpp, symmetry/su2.hpp)
 *
 * Convenience aliases (defined at bottom):
 *   using TensorU1      = BlockTensor<dcmplex, U1Symmetry>;   // was tensor_u1
 *   using TensorSU2     = BlockTensor<dcmplex, SU2Symmetry>;  // was tensor_su2 (complex)
 *   using TensorSU2Real = BlockTensor<double,  SU2Symmetry>;  // was tensor_su2 (real)
 *   using TensorU1Real  = BlockTensor<double,  U1Symmetry>;   // new
 */

#pragma once
#include "scalar.hpp"
#include "dense.hpp"
#include "symmetry/u1.hpp"
#include "symmetry/su2.hpp"

template<typename Scalar, typename Sym>
class BlockTensor {
public:
  using StructType = typename Sym::StructType;
  using BondType   = typename Sym::BondType;

private:
  int nbond;    // number of bond indices (3 for MPS site tensor)
  int nten;     // number of symmetry-allowed blocks
  int locspin;  // 2*J (SU2) or 2*Sz (U1) of the local physical degree of freedom

  Dense<Scalar>  *data_blocks;  // [nten] block data (reduced elements for SU2, full for U1)
  Dense<double>  *cgc_blocks;   // [nten] CGC tensors (only allocated when Sym::has_cgc)

  StructType cgc;  // symmetry metadata (bond quantum numbers, block sizes, etc.)

public:
  // ── Construction / destruction ───────────────────────────────────────────
  BlockTensor();
  ~BlockTensor();
  BlockTensor(const BlockTensor&);
  void clean();

  // Initialise from bond descriptors and (optionally) element data.
  void set(int nbond, int locspin, const BondType* bonds, Dense<Scalar>** blocks = nullptr);
  void set(const StructType& s, const Scalar* data, const double* cgc_data);
  void set(const StructType& s, const Scalar* data);

  // ── Accessors ────────────────────────────────────────────────────────────
  int get_nbond()   const { return nbond; }
  int get_nten()    const { return nten;  }
  int get_locspin() const { return locspin; }

  StructType&       get_struct()       { return cgc; }
  const StructType& get_struct() const { return cgc; }

  Dense<Scalar>*  get_data_block(int b)  { return &data_blocks[b]; }
  Dense<double>*  get_cgc_block(int b)   { return &cgc_blocks[b];  }  // nullptr if !has_cgc
  Dense<Scalar>*  get_parr(int b)        { return &data_blocks[b]; }  // alias for get_data_block

  void get_bond(int bond_idx, BondType& out) const { cgc.get_bond(bond_idx, out); }

  // Quantum number accessors delegated to StructType:
  int get_nmoment(int i)         const { return cgc.get_nmoment(i); }
  int get_bonddir(int i)         const { return cgc.get_bonddir(i); }
  int get_angularmoment(int i,int j)const{ return cgc.get_angularmoment(i,j); }
  int get_bonddim(int i, int j)  const { return cgc.get_bonddim(i,j); }
  int get_cgcdim(int i, int j)   const { return cgc.get_cgcdim(i,j); }

  // Block index lookup: given one moment index per bond, return block index (-1 if not allowed)
  int get_tensor_index(const int* moment_indices) const { return cgc.get_tensor_index(moment_indices); }
  int get_angularmoment_index(int bond_idx, int q) const {
    return cgc.get_angularmoment_index(bond_idx, q);
  }
  bool get_tensor_argument(int b, int* moment, int* bdim, int* cdim) const;
  bool get_tensor_argument(int b, int* moment, int* bdim) const;

  // Total number of variational parameters (data elements)
  void get_nelement(int& ndata, int& ncgc) const { cgc.get_nelement(ndata, ncgc); }

  // Flat read/write of all data (for I/O)
  void get_elems(Scalar* data_out, double* cgc_out) const;
  void get_elems(Scalar* data_out, double* cgc_out, int* block_sizes) const;
  bool is_null(int block_idx) const;
  bool is_null() const;
  bool check_angularmoments(const int* moment_idx) const;

  // ── Arithmetic ───────────────────────────────────────────────────────────
  BlockTensor& operator=(double s);
  BlockTensor& operator=(const BlockTensor& o);
  BlockTensor& operator+=(const BlockTensor& o);
  BlockTensor& operator-=(const BlockTensor& o);
  BlockTensor& operator*=(double s);
  BlockTensor& operator*=(dcmplex s);
  BlockTensor& operator/=(double s);
  bool operator==(const BlockTensor& o) const;
  bool operator!=(const BlockTensor& o) const;

  Scalar inner_prod_u(const BlockTensor& o) const; //x * y
  Scalar inner_prod_c(const BlockTensor& o) const; //x^{dagger} * y
  Scalar take_trace() const;
  double normalize_vector();

  // ── Index manipulation ────────────────────────────────────────────────────
  void shift(int bond_idx, int delta);
  void exchangeindex(int i, int j);
  BlockTensor& left2right_vectran();   // bring into mixed canonical form (left part)
  BlockTensor& right2left_vectran();   // bring into mixed canonical form (right part)

  // ── Fusing / splitting bonds ──────────────────────────────────────────────
  void fuse(int i, int j);                             // fuse adjacent bonds i,j → i
  void fuse(const BondType& a, const BondType& b);
  void fuse_to_multiplet(const BondType& a, const BondType& b, int target_j2); // SU2 only
  void fuse_to_singlet(const BondType& a, const BondType& b); // SU2 only

  // ── SVD ──────────────────────────────────────────────────────────────────
  // Returns number of singular values kept.
  void svd(BlockTensor& U, double cut_pos, BlockTensor& V, double cut_pos2, double* svals);

  void multiply_singular_value(int bond_idx, double* svals);
  void devide_singular_value(int bond_idx, double* svals);

  // ── Contraction ──────────────────────────────────────────────────────────
  // *this = contract A (on index ia) with B (on index ib)
  void contract(const BlockTensor& A, int ia, const BlockTensor& B, int ib);

  // Conjugate: flip bond directions (time reversal / hermitian conjugate)
  void take_conjugate(int bond_idx);
  void take_conjugate();

  // ── Direct sum ────────────────────────────────────────────────────────────
  void direct_sum(int bond_idx, const BlockTensor& a, const BlockTensor& b);

  // ── Printing / debugging ──────────────────────────────────────────────────
  void print() const;

  // ── Operator / local tensor construction ─────────────────────────────────
  // Spin-1/2 local operators (used by DMRG to initialise boundary operators)
  void make_spinor_start(int locspin);

  // ── DMRG: Environment update ─────────────────────────────────────────────
  // These replicate the overlap_initial / overlap_transformation / operator_*
  // methods from tensor_su2 / tensor_u1, unified here.

  // Build initial overlap block: flag=0 left boundary, flag=1 right boundary
  void overlap_initial(BlockTensor& uu, BlockTensor& vv, int flag);

  // Grow overlap block one site
  void overlap_transformation(BlockTensor& uu, BlockTensor& vv, BlockTensor& op, int flag);

  // Build initial operator block (single operator)
  void operator_initial(BlockTensor& uu, BlockTensor& vv, BlockTensor& op, int flag);

  // Grow operator block one site
  void operator_transformation(BlockTensor& uu, BlockTensor& vv, BlockTensor& op, int flag);

  // Two-operator pair-up
  void operator_pairup(BlockTensor& uu, BlockTensor& vv,
                       BlockTensor& op1, BlockTensor& op2, int flag);

  // ── DMRG: H * |v⟩ ────────────────────────────────────────────────────────
  void hamiltonian_vector_multiplication(BlockTensor& vec, BlockTensor& op1, BlockTensor& op2);

  // ── Wigner-Eckart / CGC construction (SU2 specific, no-op for U1) ────────
  void make_standard_cgc();

  // ── Two-layer helper ──────────────────────────────────────────────────────
  // Prepare wavefunction vector for Lanczos: initialise from scratch or from
  // a previous sweep's solution (used by both single-layer and two-layer DMRG)
  void makeup_input_vector();
  void initialize_input_vector();
  void random_initialize_tensor();

  // ── Check unitarity (debugging) ───────────────────────────────────────────
  void diagonalize();
};

// ── Convenience aliases ───────────────────────────────────────────────────────

using TensorU1      = BlockTensor<dcmplex, U1Symmetry>;   // was tensor_u1   (mps_u1)
using TensorSU2     = BlockTensor<dcmplex, SU2Symmetry>;  // was tensor_su2  (mps_u1, complex)
using TensorSU2Real = BlockTensor<double,  SU2Symmetry>;  // was tensor_su2  (mps_real)
using TensorU1Real  = BlockTensor<double,  U1Symmetry>;   // new

// ── Explicit instantiation declarations ──────────────────────────────────────
// Implemented in src/block_tensor_inst.cpp

// extern template class BlockTensor<double,  U1Symmetry>;   // TODO: U1 bugs pending
// extern template class BlockTensor<dcmplex, U1Symmetry>;   // TODO: U1 bugs pending
extern template class BlockTensor<double,  SU2Symmetry>;
extern template class BlockTensor<dcmplex, SU2Symmetry>;
