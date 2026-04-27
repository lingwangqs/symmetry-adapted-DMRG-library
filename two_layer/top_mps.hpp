/**
 * two_layer/top_mps.hpp — Fixed isometry (top-layer block MPS).
 *
 * In the two-layer MPS ansatz:
 *
 *   |Ψ⟩ = Σ_{m_1...m_L} A[1]^{m_1} ... A[L]^{m_L}  ·
 *           Π_i  W_i^{m_i, σ_i}  |σ_1 ... σ_N⟩
 *
 * The top layer consists of fixed isometry tensors W_i that map a patch of
 * physical spins {σ} to a single renormalised index m_i used by the bottom
 * variational MPS A[i].
 *
 * Connection to the symm_basis project
 * -------------------------------------
 * The TopMPS directly realises the "block MPS" described in the gen_mps
 * project: each W_i encodes the local patch basis constructed by
 * SymmBasis / SymmetryGroup, with symmetry labels inherited from those
 * objects.  The top-layer tensors are filled once (from a precomputed basis
 * or from a separate optimisation) and then frozen during the DMRG sweep
 * of the bottom layer.
 *
 * Template parameters
 * -------------------
 *   Scalar  = double or dcmplex
 *   Sym     = U1Symmetry or SU2Symmetry
 */

#pragma once
#include "../block_tensor.hpp"

template<typename Scalar, typename Sym>
class TopMPS {
  using TensorType = BlockTensor<Scalar, Sym>;
  using BondType   = typename Sym::BondType;

private:
  int nsites;       // number of super-sites (= number of patches)
  int patch_size;   // number of physical spins per patch (e.g. 4 for 2×2)
  int phys_dim;     // physical dimension per patch (d^patch_size)
  int renorm_dim;   // renormalised (output) dimension of each isometry

  TensorType *W;    // [nsites] isometry tensors W_i
                    //   index structure: [left_bond, phys_patch, right_bond]
                    //   W_i W_i† = 1  (isometry property)

  BondType   *virtual_bonds; // [nsites+1] bond quantum numbers between super-sites
  BondType   *phys_bonds;    // [nsites]   patch physical bond (fused from patch_size spins)

public:
  TopMPS();
  ~TopMPS();

  // ── Initialisation ────────────────────────────────────────────────────────

  // Allocate and zero all isometry tensors.
  void allocate(int nsites, int patch_size, int phys_dim, int renorm_dim,
                const BondType* vbonds, const BondType* pbonds);

  // Load pre-computed isometries (e.g. from symm_basis output).
  // data[i] is the flattened block data for W_i in the block-tensor layout.
  void load(int site, const Scalar* data, const double* cgc_data, int ndata, int ncgc);

  // Read/write from disk (same format as DMRG::save_mps2)
  void save(const char* prefix) const;
  bool load(const char* prefix);

  // ── Isometry access ───────────────────────────────────────────────────────
  TensorType& get_W(int site)       { return W[site]; }
  const TensorType& get_W(int site) const { return W[site]; }

  int get_nsites()     const { return nsites; }
  int get_patch_size() const { return patch_size; }
  int get_phys_dim()   const { return phys_dim; }
  int get_renorm_dim() const { return renorm_dim; }

  // Virtual bond between site i and i+1
  const BondType& get_virtual_bond(int i) const { return virtual_bonds[i]; }
  // Physical (patch) bond at site i
  const BondType& get_phys_bond(int i)    const { return phys_bonds[i]; }

  // ── Isometry check ────────────────────────────────────────────────────────
  // Verify W_i W_i† = 1 at all sites (up to tol).
  bool check_isometry(double tol = 1e-10) const;

  // ── Renormalised operator construction ────────────────────────────────────
  // Project a physical operator O (acting on the patch at site i) through W_i:
  //   O_renorm = W_i† · O · W_i
  // Used to build the effective Hamiltonian in the renormalised basis.
  void project_operator(int site, const TensorType& O_phys,
                        TensorType& O_renorm) const;

  // Project a two-site operator O12 acting on patches (i, i+1):
  //   O_renorm = (W_i ⊗ W_{i+1})† · O12 · (W_i ⊗ W_{i+1})
  void project_two_site_operator(int site_l,
                                  const TensorType& O12_phys,
                                  TensorType& O_renorm) const;

  // ── Environment update (called from TwoLayerDMRG::sweep) ─────────────────
  // Build the effective left/right environment at position site,
  // combining the isometry W_i with the bottom MPS tensor A[i].
  //
  //   left_env[site+1] = contract(left_env[site], W[site], A[site], W†[site])
  void update_left_env(int site, const TensorType& bottom_A,
                        const TensorType& old_left_env,
                        TensorType& new_left_env) const;

  void update_right_env(int site, const TensorType& bottom_A,
                         const TensorType& old_right_env,
                         TensorType& new_right_env) const;

  // ── Hamiltonian environment (for H*v in bottom layer) ─────────────────────
  // Build the effective Hamiltonian environment block at site, combining
  // the physical Hamiltonian (in patch form) with the isometry.
  //
  // ham_phys[site] is the MPO tensor of the Hamiltonian in physical (patch) space.
  // ham_renorm[site] is the result: MPO tensor in renormalised space.
  void renormalise_hamiltonian_op(int site, const TensorType& ham_phys,
                                   TensorType& ham_renorm) const;
};

// ── Explicit instantiation declarations ──────────────────────────────────────
extern template class TopMPS<double,  SU2Symmetry>;
extern template class TopMPS<dcmplex, SU2Symmetry>;
extern template class TopMPS<dcmplex, U1Symmetry>;
