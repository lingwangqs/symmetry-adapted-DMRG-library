/**
 * two_layer/bottom_mps.hpp — Variational MPS (bottom layer).
 *
 * The bottom MPS A[i] operates in the renormalised single-site Hilbert
 * space defined by the top-layer isometry W_i.  Its physical index m_i
 * runs over the renormalised states output by W_i (dimension = renorm_dim).
 *
 * During a DMRG sweep, the two-site effective Hamiltonian is constructed
 * from:
 *   left_env[i]   — left environment block in renormalised space
 *   right_env[i]  — right environment block in renormalised space
 *   ham_renorm[i] — MPO tensor (or pair of tensors) in renormalised space
 *
 * All three are computed using TopMPS::update_*_env and
 * TopMPS::renormalise_hamiltonian_op, so this class only needs to handle
 * the variational optimisation of A[i] and canonical form maintenance.
 *
 * Template parameters: same as TopMPS (Scalar, Sym).
 */

#pragma once
#include "../block_tensor.hpp"
#include "../lanczos.hpp"

template<typename Scalar, typename Sym>
class BottomMPS {
  using TensorType  = BlockTensor<Scalar, Sym>;
  using LanczosType = Lanczos<Scalar, Sym>;
  using BondType    = typename Sym::BondType;

private:
  int nsites;      // number of renormalised sites
  int bondd;       // maximum bond dimension
  int max_exci;    // number of states to target

  TensorType *A;         // [nsites] variational MPS tensors
  double     **sv;       // [max_exci][nsites] singular value arrays (for truncation)
  double     gs_enr[100]; // ground-state energy per excitation index
  double     err;         // current truncation error
  int        exci;        // current excitation index
  int        xleft, xrght; // canonical centre position

  // Orthogonality projectors (for excited states)
  TensorType **ovlp;    // [nsites][max_exci] overlap environment blocks
  TensorType **orth;    // [nsites][max_exci] stored excited-state vectors
  TensorType *overlapvec; // [max_exci]

public:
  BottomMPS();
  ~BottomMPS();

  // ── Initialisation ────────────────────────────────────────────────────────
  void allocate(int nsites, int bondd, int max_exci, const BondType* vbonds);
  void random_init();
  void make_left_canonical_form();

  // ── Accessors ─────────────────────────────────────────────────────────────
  TensorType&       get_A(int site)       { return A[site]; }
  const TensorType& get_A(int site) const { return A[site]; }
  int  get_nsites()   const { return nsites; }
  int  get_bondd()    const { return bondd; }
  double get_enr()    const { return gs_enr[exci]; }
  double get_error()  const { return err; }

  // ── Canonical form ─────────────────────────────────────────────────────────
  // Move canonical centre from current position to `target_site`.
  void shift_canonical_centre(int target_site, const TensorType* left_envs,
                               const TensorType* right_envs);

  // ── Two-site optimisation step (called from TwoLayerDMRG::sweep) ──────────
  // Solve H_eff |v⟩ = E |v⟩ for the two-site tensor at (site_l, site_r),
  // then split via SVD and update A[site_l], A[site_r], and sv[exci][site_l].
  //
  // Parameters:
  //   left_env   — effective left environment block at site_l
  //   right_env  — effective right environment block at site_r
  //   ham_l      — renormalised Hamiltonian MPO tensor at site_l
  //   ham_r      — renormalised Hamiltonian MPO tensor at site_r
  //   dir        — 0 (left→right), 1 (right→left)
  void optimize_two_site(int site_l, int site_r,
                          const TensorType& left_env,
                          const TensorType& right_env,
                          const TensorType& ham_l,
                          const TensorType& ham_r,
                          int dir, int n_already_converged);

  // ── Measurement ───────────────────────────────────────────────────────────
  void mea_enr();
  void mea_enr(int exci_idx, double* out);

  // ── I/O ──────────────────────────────────────────────────────────────────
  void save(const char* prefix, int n_sweeps = 0) const;
  bool load(const char* prefix, int n_sweeps = 0);
  void save_sv(const char* prefix)  const;
  void load_sv(const char* prefix);
};

// ── Explicit instantiation declarations ──────────────────────────────────────
extern template class BottomMPS<double,  SU2Symmetry>;
extern template class BottomMPS<dcmplex, SU2Symmetry>;
extern template class BottomMPS<dcmplex, U1Symmetry>;
