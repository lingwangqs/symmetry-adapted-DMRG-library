/**
 * two_layer/two_layer_dmrg.hpp — DMRG optimisation of the two-layer MPS.
 *
 * Structure
 * ---------
 *
 *   physical sites:  σ_1 σ_2 σ_3 σ_4 | σ_5 σ_6 σ_7 σ_8 | ...
 *                    └─── patch 1 ───┘  └─── patch 2 ───┘
 *                          │                    │
 *   top isometry:    W_1 (fixed)          W_2 (fixed)
 *                          │                    │
 *   renorm. sites:   m_1                  m_2
 *                    │                    │
 *   bottom MPS:  ── A_1 ────────────── A_2 ──────── ...
 *
 * The top-layer isometries W_i are fixed (loaded from TopMPS or from a
 * precomputed symm_basis block MPS).
 *
 * The DMRG sweep optimises the bottom-layer tensors A_i while holding W_i
 * fixed.  The effective Hamiltonian at each two-site step is built by:
 *
 *   1. Renormalising the physical Hamiltonian through W_i → H_renorm[i]
 *   2. Building left/right environments in the renormalised space via
 *      TopMPS::update_{left,right}_env
 *   3. Solving the 2-site Lanczos problem for A_i, A_{i+1}
 *   4. Splitting via SVD with bond dimension truncation
 *
 * Future extension: alternating optimisation where W_i is also updated in
 * a separate sweep (using the bottom MPS as the environment).
 *
 * Template parameters: same as TopMPS / BottomMPS.
 */

#pragma once
#include "top_mps.hpp"
#include "bottom_mps.hpp"
#include "../block_tensor.hpp"

template<typename Scalar, typename Sym>
class TwoLayerDMRG {
  using TensorType = BlockTensor<Scalar, Sym>;
  using BondType   = typename Sym::BondType;

private:
  // ── Layers ────────────────────────────────────────────────────────────────
  TopMPS<Scalar, Sym>    top;    // fixed isometry
  BottomMPS<Scalar, Sym> bottom; // variational

  // ── Hamiltonian ───────────────────────────────────────────────────────────
  // Physical Hamiltonian MPO tensors (in patch space)
  TensorType *ham_phys;      // [nsites]   physical MPO at each super-site
  TensorType *ham_renorm;    // [nsites]   renormalised MPO: W† H_phys W

  // Coupling constants (e.g. J1, J2 per bond in patch)
  int      nbonds;   // number of coupling terms
  double  *coup;     // [nbonds] coupling constants
  int    **hmap;     // [nsites][nbonds] Hamiltonian bond index map

  // ── Environments ──────────────────────────────────────────────────────────
  // left_env[i]: effective environment block to the left of super-site i
  // right_env[i]: effective environment block to the right of super-site i
  TensorType *left_env;   // [nsites+1]
  TensorType *right_env;  // [nsites+1]

  // Operator environments for each Hamiltonian bond term
  TensorType **opr_left;  // [nsites][nbonds]
  TensorType **opr_right; // [nsites][nbonds]

  // ── State ─────────────────────────────────────────────────────────────────
  int nsites;    // number of super-sites
  int sweep_idx; // number of completed sweeps

  // ── Private methods ───────────────────────────────────────────────────────

  // Build all renormalised Hamiltonian tensors (called once after loading top)
  void build_renorm_hamiltonian();

  // Build the initial left/right environments from scratch
  void build_initial_environments();

  // Left sweep: move canonical centre from right to left, optimising each pair
  void sweep_to_left();

  // Right sweep: move canonical centre from left to right, optimising each pair
  void sweep_to_right();

  // Update left environment one step to the right
  void grow_left_env(int site);

  // Update right environment one step to the left
  void grow_right_env(int site);

public:
  TwoLayerDMRG();
  ~TwoLayerDMRG();

  // ── Setup ─────────────────────────────────────────────────────────────────

  // Load a pre-computed TopMPS (e.g. from symm_basis block MPS output)
  void load_top(const TopMPS<Scalar,Sym>& top_in) { top = top_in; }

  // Set the physical Hamiltonian MPO in patch space
  void set_hamiltonian(int nbonds, const double* couplings, int** bond_map,
                        const TensorType* ham_phys_in);

  // Initialise the bottom MPS randomly with given bond dimension
  void init_bottom_random(int bondd);

  // Initialise the bottom MPS from a previously saved file
  void init_bottom_from_file(const char* prefix);

  // ── Main DMRG interface ───────────────────────────────────────────────────

  // Perform n_sweeps full left-right sweeps of the bottom MPS.
  void run(int n_sweeps);

  // Single sweep (for fine-grained control)
  void sweep();

  // ── Results ───────────────────────────────────────────────────────────────
  double get_energy()      const { return bottom.get_enr(); }
  double get_trunc_error() const { return bottom.get_error(); }
  int    get_nsweeps()     const { return sweep_idx; }

  const BottomMPS<Scalar,Sym>& get_bottom() const { return bottom; }
  const TopMPS<Scalar,Sym>&    get_top()    const { return top;    }

  // ── I/O ──────────────────────────────────────────────────────────────────
  void save(const char* prefix) const;
  bool load(const char* prefix);

  // ── Future: alternating optimisation (top + bottom) ───────────────────────
  // void sweep_top(int n_sweeps);   // optimise top while fixing bottom
};

// ── Explicit instantiation declarations ──────────────────────────────────────
extern template class TwoLayerDMRG<double,  SU2Symmetry>;
extern template class TwoLayerDMRG<dcmplex, SU2Symmetry>;
extern template class TwoLayerDMRG<dcmplex, U1Symmetry>;
