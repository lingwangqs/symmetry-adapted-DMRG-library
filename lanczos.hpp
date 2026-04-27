/**
 * lanczos.hpp — Lanczos eigensolver for symmetric (Hermitian) problems.
 *
 * Replaces:
 *   lanczos_su2.hpp / .cpp  (mps_real, mps_u1)  — Lanczos<Scalar, SU2Symmetry>
 *   lanczos_u1.hpp  / .cpp  (mps_u1)             — Lanczos<Scalar, U1Symmetry>
 *
 * The Lanczos class finds the lowest few eigenvalues and eigenvectors of
 * the effective Hamiltonian H_eff at the current DMRG site, stored as a
 * BlockTensor<Scalar, Sym>.
 *
 * Three Lanczos variants are provided (matching the original code):
 *   lanczos1  — basic Krylov with no re-orthogonalisation
 *   lanczos2  — selective re-orthogonalisation (for multiple targets)
 *   lanczos3  — full re-orthogonalisation against all Krylov vectors
 *
 * Template instantiations (compiled in src/lanczos_inst.cpp):
 *   Lanczos<double,  SU2Symmetry>
 *   Lanczos<dcmplex, SU2Symmetry>
 *   Lanczos<dcmplex, U1Symmetry>
 */

#pragma once
#include "block_tensor.hpp"

template<typename Scalar, typename Sym>
class Lanczos {
  using TensorType = BlockTensor<Scalar, Sym>;
  using BondType   = typename Sym::BondType;

private:
  int     nrep;    // maximum number of Lanczos steps
  int     mlanc;   // current Krylov subspace dimension
  int     neig;    // number of eigenstates to target
  double *aal;     // diagonal elements a[0..mlanc-1]
  double *nnl;     // off-diagonal elements n[0..mlanc-1]
  double *eig;     // eigenvalues [neig]
  double *vec;     // eigenvectors of tridiagonal matrix [mlanc * mlanc]
  double  enrexp;  // expected energy (used as shift for excited states)

  TensorType *ff;  // Krylov basis vectors [nrep]

  // Diagonalise the accumulated mlanc×mlanc tridiagonal matrix (LAPACK dstev)
  void diatridiag(int mlanc);

  // Check eigenvector quality: compute |H|v> - E|v>|  for state k
  void check_eigenvector(int site_l, int site_r, TensorType& vec,
                         double& res_norm, double& energy, int state_idx);

public:
  Lanczos();
  ~Lanczos();

  // Allocate Krylov space and set expected energy (for shift-invert or excited)
  void initialize_lanczos(TensorType& v0, int nrep, int neig);
  void set_expected_energy(double e) { enrexp = e; }

  // ── Single eigenvector (ground state) ────────────────────────────────────

  // lanczos1: standard Lanczos, no re-orthogonalisation
  // (fast, may lose orthogonality after many steps)
  void lanczos1(int site_l, int site_r, TensorType& result);

  // lanczos2: selective re-orthogonalisation against already-converged states
  // (used for excited states: neig > 1)
  void lanczos2(int site_l, int site_r, TensorType& result, int n_already_converged);

  // lanczos3: full re-orthogonalisation (most stable, most expensive)
  void lanczos3(int site_l, int site_r, TensorType& result, int n_already_converged);

  // ── Multiple eigenvectors ─────────────────────────────────────────────────

  // Solve for a single Ritz eigenvector after diagonalising the tridiagonal matrix
  // Called after lanczos1/2/3 to extract the desired state.
  void compute_eigenvector(int site_l, int site_r, TensorType& result,
                           int state_idx, int& n_steps, int which_lanczos);
  void compute_eigenvector(TensorType& result, int state_idx);
  void compute_eigenvector(TensorType& result, int state_idx, int which_coeff);

  // ── Imaginary time evolution (beta-double / iTEBD helper) ────────────────
  void compute_evolution(TensorType& result, int site_l, int site_r);
  void compute_evolution(TensorType& result, int site_l, int site_r, int state_idx);

  // ── Full 2-site diagonalisation (called from DMRG::sweep) ────────────────
  // diag_op performs the full Lanczos solve and updates result.
  void diag_op(int site_l, int site_r, TensorType& result, TensorType& h_vec, int n_conv);

  // ── Accessors ─────────────────────────────────────────────────────────────
  double get_eigval()         const { return eig[0]; }
  double get_eigval(int k)    const { return eig[k]; }
};

// ── Explicit instantiation declarations ──────────────────────────────────────

extern template class Lanczos<double,  SU2Symmetry>;
extern template class Lanczos<dcmplex, SU2Symmetry>;
extern template class Lanczos<dcmplex, U1Symmetry>;
