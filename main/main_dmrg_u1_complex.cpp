/**
 * main/main_dmrg_u1_complex.cpp
 *
 * Test driver for DMRG<dcmplex, U1Symmetry> on a 1D Heisenberg chain.
 *
 * Usage:
 *   ./dmrg_u1_complex -lx <L> -d <D> -sec <Sz_sector> -niter <niter>
 *                     [-dr <D_read>] [-exci <n_exci>] [-psize <n_threads>]
 *
 * Example (spin-1/2 chain L=8, D=64, Sz=0 sector, 10 sweeps):
 *   ./dmrg_u1_complex -lx 8 -d 64 -sec 0 -niter 10
 */

#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "../dmrg.hpp"
#include "../cgc_tables.hpp"

using namespace std;

// ── Global DMRG chain pointer (used by Lanczos) ───────────────────────────────
DMRG<dcmplex, U1Symmetry>* chain_ptr = nullptr;
// Global parameters referenced by block_tensor_inst.cpp and dmrg_inst.cpp via extern
int        max_dcut     = 64;
int        psize        = 1;
int        myrank       = 0;   // MPI rank (single process); referenced b
CGCTables* g_cgc_tables = nullptr;

int main(int argc, char** argv) {
  // ── Defaults ──
  int lx     = 4;     // chain length
  int ly     = 4;     // irrelevant (1D)
  int bondd  = 64;    // max bond dimension D
  int sec    = 0;     // total Sz sector (2*Sz for U1)
  int exci   = 0;     // number of excited states - 1 (0 = ground state only)
  int niter  = 2;    // number of DMRG sweeps
  int read   = 0;     // read existing MPS from disk (0 = start from scratch)
  int read2  = 0;     // bond dimension of pre-existing MPS to read

  // ── Parse arguments ──
  for (int i = 1; i < argc; ++i) {
    if      (strcmp(argv[i], "-lx")    == 0) lx    = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-ly")    == 0) ly    = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-d")     == 0) bondd = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-sec")   == 0) sec   = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-exci")  == 0) exci  = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-niter") == 0) niter = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-psize") == 0) psize = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-dr")    == 0) read  = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-dge")   == 0) read2 = atoi(argv[i+1]);
  }

  omp_set_num_threads(psize);

  cout << "U1 DMRG: lx=" << lx << " ly=" << ly << " D=" << bondd
       << " sec=" << sec << " exci=" << exci << " niter=" << niter
       << " threads=" << psize << endl;

  // ── phdim for spin-1/2: 2 states per site, physpn = 1 (2*S=1) ──
  int physpn = 1;   // U1: physical dimension = 2S+1 = 2 for spin-1/2

  DMRG<dcmplex, U1Symmetry> chain(ly, lx, bondd, physpn, sec, exci);
  chain_ptr = &chain;

  bool pass = false;
  if (read > 0) {
    pass = chain.read_mps(read, read2);
    cout << "read_mps pass=" << pass << endl;
    if (pass) {
      chain.read_enr(read, read2);
      chain.read_ww(read, read2);
      chain.prepare_sweep();
    }
  }

  if (!pass) {
    chain.do_idmrg();
  }

  double enr1 = chain.get_enr();
  double enr2 = enr1;
  for (int i = 0; i < niter; ++i) {
    chain.sweep();
    enr2 = chain.get_enr();
    cout << setprecision(12) << "sweep=" << i
         << "  E=" << enr2
         << "  dE=" << fabs(enr1 - enr2) << endl;
    if (fabs(enr1 - enr2) < 1e-6) {
      cout << "Energy converged after " << i+1 << " sweeps." << endl;
      break;
    }
    enr1 = enr2;
  }

  cout << "Final energy: " << setprecision(14) << chain.get_enr() << endl;
  // For a 1D Heisenberg chain of length L, exact ground state energy per bond
  // approaches -ln(2) + 1/4 ≈ -0.4431 per site in the thermodynamic limit.
  cout << "E/bond ≈ " << chain.get_enr() / (lx * ly) << endl;

  return 0;
}
