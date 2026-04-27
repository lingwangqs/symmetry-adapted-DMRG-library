/**
 * dmrg.hpp — Two-site DMRG, templated on scalar type and symmetry.
 *
 * Replaces:
 *   dmrg_su2_omp.hpp / .cpp   (mps_u1)   — DMRG<dcmplex, SU2Symmetry>
 *   dmrg_u1_omp.hpp  / .cpp   (mps_u1)   — DMRG<dcmplex, U1Symmetry>
 *   dmrg_su2_mpi.hpp / .cpp   (mps_real) — DMRG<double,  SU2Symmetry>
 *
 * The original dmrg_su2 and dmrg_u1 classes are byte-for-byte identical
 * except for the BlockTensor specialisation they operate on.  They are
 * unified here into a single template class.
 *
 * Parallelism
 * -----------
 * OMP parallelism is applied inside the environment-update and H*v routines
 * via OpenMP pragmas in the implementation.  MPI is an optional extension
 * (see main/main_dmrg_mpi.cpp for the MPI launch wrapper).
 *
 * Template instantiations (src/dmrg_inst.cpp):
 *   DMRG<double,  SU2Symmetry>  — real SU2 (mps_real)
 *   DMRG<dcmplex, SU2Symmetry>  — complex SU2 (mps_u1/mps_complex)
 *   DMRG<dcmplex, U1Symmetry>   — complex U1  (mps_u1)
 */

#pragma once
#include "block_tensor.hpp"
#include "lanczos.hpp"

template<typename Scalar, typename Sym>
class DMRG {
  using TensorType  = BlockTensor<Scalar, Sym>;
  using LanczosType = Lanczos<Scalar, Sym>;
  using BondType    = typename Sym::BondType;
  using StructType  = typename Sym::StructType;

private:
  // ── Geometry ─────────────────────────────────────────────────────────────
  int ly, lx;     // lattice dimensions (physical)
  int ns;          // total number of MPS sites
  int nfree;       // number of frozen boundary sites

  // ── Bond dimension & excitations ─────────────────────────────────────────
  int bondd;       // maximum bond dimension (D_cut)
  int exci;        // current excitation index (0 = ground state)
  int max_exci;    // number of eigenstates to target
  int totspin;     // 2 * total spin quantum number
  int phdim;       // physical Hilbert space dimension per site
  int physpn;      // physical spin value (2*S)
  int nevolution;  // number of imaginary-time evolution steps
  int xleft, xrght; // current canonical center position
  int badposition; // site where canonical form last broke down

  // ── DMRG state ────────────────────────────────────────────────────────────
  int **fll;   // [ns][nbb] — left Hamiltonian bond map
  int **frr;   // [ns][nbb] — right Hamiltonian bond map
  int **hmap;  // [ns][nbb] — Hamiltonian bond index map

  TensorType *uu;      // [ns]    MPS site tensors
  TensorType *vv;      // [ns]    auxiliary MPS (for excited states / compression)
  TensorType *hh;      // [ns]    effective Hamiltonian at each site
  TensorType *dltu;    // [ns]    difference tensors (compression error)
  TensorType *comp;    // [ns]    compression target MPS
  TensorType **opr;    // [ns][nbb] left/right operator environment blocks
  TensorType **ovlp;   // [ns][max_exci] overlap environment (for orthogonality)
  TensorType **orth;   // [ns][max_exci] stored excited-state vectors
  TensorType **tran;   // [ns][max_exci] translation operator environments
  TensorType *overlapvec; // [max_exci] overlap with stored excited states

  TensorType sigma[4]; // local spin operators: sigma+, sigma-, sigmaz, identity
  TensorType ring[8];  // ring-exchange operators (for J1-J2 / plaquette models)
  TensorType spnswap;  // spin-swap permutation operator
  TensorType spnproj;  // spin projection operator

  Dense<Scalar> **step2; // [ns][nbb] scratch tensors for 5-step operator update

  // ── Energies / truncation ─────────────────────────────────────────────────
  double gs_enr[100];    // ground-state energy per excitation index
  double enr[100];       // current energy estimate per excitation index
  double *coup;          // [nbb] coupling constants (J1, J2, ...)
  double err;            // truncation error
  double **ww;           // [max_exci][ns] singular value arrays
  double *wtmp;          // scratch singular values
  double *bond_enr;      // [ns-1] bond energy at each link
  double *bond_enr_all;  // [ns-1] accumulated bond energies
  double wt;             // total weight of discarded singular values
  double vecnorm;        // norm of current wavefunction vector

  // ── Private methods ───────────────────────────────────────────────────────
  void initialize_local_operators();
  void lanczos_solve_eigenvector_idmrg(int site_l, int site_r, TensorType& result);
  void prepare_site_operator_from_left(int site, int exci_idx);
  void prepare_site_operator_from_right(int site, int exci_idx);
  void prepare_site_operator_from_left(int site);
  void prepare_site_operator_from_right(int site);

public:
  // ── Constructors ─────────────────────────────────────────────────────────
  DMRG();
  DMRG(int ly, int lx, int bondd, int phdim, int totspin, int max_exci);
  ~DMRG();

  double get_enr() const { return gs_enr[exci]; }

  // ── DMRG algorithms ───────────────────────────────────────────────────────

  // Infinite-DMRG: grow chain from 2 sites to ns sites
  void do_idmrg();

  // Finite-DMRG sweeps
  void prepare_sweep();  // initialise environments from scratch
  void sweep();          // one full left-right-left sweep

  // ── H * |v⟩ for iDMRG ────────────────────────────────────────────────────
  void hamiltonian_vector_multiplication_idmrg(int site_l, int site_r,
                                                TensorType& v_in, TensorType& v_out);

  // ── Canonical form utilities ──────────────────────────────────────────────
  void wavefunc_transformation(int site, int dir);  // move canonical centre

  // ── Input vector preparation for Lanczos ─────────────────────────────────
  void prepare_input_vector(int site_l, int site_r, TensorType& v0);
  void prepare_input_vector_initial(int site_l, int site_r, TensorType& v0);

  // ── I/O ──────────────────────────────────────────────────────────────────
  void save_enr();
  void read_enr(int n_sweeps, int exci_idx);
  bool read_mps(int n_sweeps, int exci_idx, int which);
  bool read_mps(int n_sweeps, int exci_idx);
  void save_mps2(int exci);
  void save_ww();
  void read_ww(int n_sweeps, int exci_idx);

};

// ── Explicit instantiation declarations ──────────────────────────────────────

extern template class DMRG<double,  SU2Symmetry>;
extern template class DMRG<dcmplex, SU2Symmetry>;
// extern template class DMRG<dcmplex, U1Symmetry>;  // TODO: U1 bugs pending
