/**
 * dense.hpp — Dense rank-N tensor, templated on scalar type.
 *
 * Replaces:
 *   tensor.hpp / tensor.cpp        (mps_real, mps_u1) — Scalar = double
 *   tensor_dcmplex.hpp / .cpp      (mps_u1)            — Scalar = dcmplex
 *
 * Dense<Scalar> stores a flattened C-order array of Scalar elements and
 * provides the contraction, SVD, eigenvalue, and DMRG-environment update
 * operations used throughout the DMRG algorithm.
 *
 * All numerical routines dispatch to ScalarTraits<Scalar>::gemm / heev /
 * gesvd (see scalar.hpp), so both real and complex are handled identically
 * at the call site.
 *
 * Template instantiations (compiled in dense_inst.cpp):
 *   Dense<double>    — real dense tensor (was class tensor)
 *   Dense<dcmplex>   — complex dense tensor (was class tensor_dcmplex)
 */

#pragma once
#include "scalar.hpp"
#include <cstring>

template<typename Scalar>
class Dense {
private:
  int   nbond;    // number of indices
  int   nelem;    // total number of elements = product of bonddim[i]
  int  *bonddim;  // dimensions of each index [nbond]
  Scalar *data;   // flattened element array [nelem], C order

public:
  // ── Construction / destruction ───────────────────────────────────────────
  Dense();
  Dense(int nbond, const int* dims);
  Dense(int nbond, const int* dims, const Scalar* init);
  Dense(const Dense& o);
  ~Dense();

  void clean();
  void alloc(int nbond, const int* dims);
  void copy(int nbond, const int* dims, const Scalar* src);
  void copy(const Dense& o);
  void random_init();
  void zero();

  // ── Accessors ────────────────────────────────────────────────────────────
  int     get_nbond()       const { return nbond; }
  int     get_nelem()       const { return nelem; }
  int     get_bonddim(int i)const { return bonddim[i]; }
  int*    get_bonddim_ptr()       { return bonddim; }
  Scalar* getptr()                { return data; }
  Scalar* getptr(int offset)      { return data + offset; }
  Scalar  get_elem(int i)   const { return data[i]; }
  void    get_elems(Scalar* dst)  const { std::memcpy(dst, data, nelem * sizeof(Scalar)); }
  void    set_elems(const Scalar* src)  { std::memcpy(data, src, nelem * sizeof(Scalar)); }
  void    set_elem(int i, int j, int k, Scalar val); // 3-index access
  void    print()           const;

  // ── Arithmetic ───────────────────────────────────────────────────────────
  Dense& operator=(Scalar s);
  Dense& operator=(const Dense& o);
  Dense& operator+=(const Dense& o);
  Dense& operator-=(const Dense& o);
  Dense& operator*=(double s);
  Dense& operator*=(dcmplex s);
  Dense& operator/=(double s);
  Dense  operator+(const Dense& o) const;
  Dense  operator-(const Dense& o) const;
  bool   operator==(const Dense& o) const;
  bool   operator!=(const Dense& o) const;

  Dense& conjugate();
  double get_norm()  const;
  double rescale();       // normalise to unit norm, return old norm
  void   rescale(double); // multiply by given factor

  // ── Index manipulation ────────────────────────────────────────────────────
  void exchangeindex(int i, int j);  // transpose two indices
  void mergeindex(int i, int j);     // fuse adjacent indices i and j into i
  void separateindex(int i, int d0, int d1); // split index i into (d0, d1)
  void shift(int bond_idx, int delta);       // cyclic shift of one index

  // ── Contraction ──────────────────────────────────────────────────────────
  // *this = A contracted on index ia with B contracted on index ib
  Dense& contract(const Dense& A, int ia, const Dense& B, int ib);
  Dense& contract_v2(const Dense& A, int ia, const Dense& B, int ib);
  // Contract a single index with itself (trace over that index)
  Dense& contractindex(int i0, int i1);

  // ── SVD ──────────────────────────────────────────────────────────────────
  // Split *this across bond cut_bond → (U, svals, V)
  // cut = 0: U absorbs left indices, V absorbs right+singular values
  // cut = 1: U absorbs left+singular values, V absorbs right indices
  // n0: number of bonds grouped into the left factor (U).
  // max_kept=0 means no bond-dimension cap (kept is output only).
  void svd(Dense& U, double p1, Dense& V, double p2,
           double* svals, int& kept, int max_kept, int n0);

  void multiply_singular_value(int bond_idx, const double* svals);
  void devide_singular_value(int bond_idx, const double* svals);

  // ── Eigenvalue ───────────────────────────────────────────────────────────
  // Solve A*x = lambda*x for Hermitian/symmetric *this (overwrites with eigvecs)
  void hermitian_eig(double* evals);
  void hermitian_eig_inv();    // in-place inverse (via eigendecomposition)

  // ── Inner products ───────────────────────────────────────────────────────
  Scalar inner_prod_c(const Dense& o) const;  // sum_i conj(data[i]) * o.data[i]
  Scalar inner_prod_u(const Dense& o) const;  // sum_i data[i] * o.data[i]
  Scalar take_trace()               const;  // trace over first two equal-dim indices
  void   calculate_difference(const Dense& o, double& abs_diff, double& rel_diff) const;

  // ── DMRG environment contractions ────────────────────────────────────────
  // These match the contract_dmrg_* methods in the original tensor / tensor_dcmplex.
  // Site index convention:  dir=0 → left sweep,  dir=1 → right sweep.
  // ── CGC tensor contractions (ported from tensor_dmrg_src.cpp) ────────────
  // Used by makeup_clebsch_gordan_coefficient_tensors to build CGC factor tables.
  Dense& contract_dmrg_overlap_initial(Dense& t1, Dense& t2, int flag);
  Dense& contract_dmrg_operator_initial(Dense& t1, Dense& t2, Dense& t3, int flag);
  Dense& contract_dmrg_operator_transformation(Dense& t1, Dense& t2, Dense& t3, int flag);
  Dense& contract_dmrg_operator_pairup(Dense& uu, Dense& vv, Dense& op1, Dense& op2, int flag);
  Dense& contract_dmrg_permutation(Dense& uu, Dense& vv, Dense& vec, Dense& op1, Dense& op2, int flag);
  Dense& contract_dmrg_hamiltonian_vector_multiplication(Dense& t1, Dense& t2, int flag);
  Dense& contract_dmrg_operator_transformation_step5(const Dense& uuop, const Dense& vv, int nb, const int* bdim, int flag);//intermediate step contract UU+OP with VV
  Dense& contract_dmrg_operator_transformation_step1(const Dense& uuop, const Dense& vv, int nb, const int* bdim, int flag);//intermediate step contract UU+OP1+OP2 with VV

  // ── CGC / operator construction (SU2 helpers, used from Dense<double>) ──
  void make_cgc(int j1, int j2, int j3);
  void make_singlet(int j);
  void make_identity(int j);
  void multiply_cgc(int j1, int j2, int j3, int which_bond, int dir);
  void shift_set_identity(int dim, int nshift, const int* dims);
  void shift_copy(int i0, int i1, int i2, const Dense& src);
  void shift_copy(int i0, int i1, int i2, int j0, int j1, int j2, const Dense& src);

  // ── Predicates ───────────────────────────────────────────────────────────
  bool check_null()         const;
  bool is_identity()        const;
  bool is_minus_identity()  const;
  bool is_null()            const;
  bool is_zero()            const;
  bool is_one()             const;
  bool is_minus_one()       const;
  bool is_proportional_to(const Dense& o, Scalar& ratio);

  // ── Direct sum / tensor product ───────────────────────────────────────────
  void direct_sum(int bond_idx, const Dense& a, const Dense& b);
  void direct_subtract(int bond_idx, int dim_kept);
  void direct_product(const Dense& a, const Dense& b); // Kronecker product
  void tensor_product(const Dense& a, const Dense& b);
};

// ── Explicit instantiation declarations ──────────────────────────────────────
// Implemented in src/dense_inst.cpp

extern template class Dense<double>;
extern template class Dense<dcmplex>;
