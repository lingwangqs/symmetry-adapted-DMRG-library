/**
 * main/main_dmrg_su2_real.cpp
 *
 * Test driver for DMRG<double, SU2Symmetry> on a 1D Heisenberg chain.
 * SU2 symmetry labels blocks by total spin J (stored as 2J).
 * Uses real (double) wavefunctions — suitable for time-reversal-invariant
 * Hamiltonians with no complex hopping.
 *
 * Usage:
 *   ./dmrg_su2_real -lx <L> -d <D> -sec <2J> -niter <niter>
 *                   [-dr <D_read>] [-exci <n_exci>] [-psize <n_threads>]
 *
 * Example (spin-1/2 chain L=8, D=64, singlet sector 2J=0, 10 sweeps):
 *   ./dmrg_su2_real -lx 8 -d 64 -sec 0 -niter 10
 */

#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "../dmrg.hpp"
#include "../cgc_tables.hpp"

using namespace std;

DMRG<double, SU2Symmetry>* chain_ptr = nullptr;

// Global parameters referenced by block_tensor_inst.cpp and dmrg_inst.cpp via extern
int        max_dcut     = 64;
int        psize        = 1;
int        myrank       = 0;   // MPI rank (single process); referenced by old tensor.cpp
CGCTables* g_cgc_tables = nullptr;

int main(int argc, char** argv) {
  // ── Defaults ──
  int lx     = 4;     // chain length (ns = lx * ly)
  int ly     = 4;
  int bondd  = 64;    // max bond dimension D
  int sec    = 0;     // total spin sector (2*J; sec=0 → singlet)
  int exci   = 0;     // 0 = ground state only
  int niter  = 2;    // DMRG sweeps
  int read   = 64;
  int read2  = 0;

  for (int i = 1; i < argc; ++i) {
    if      (strcmp(argv[i], "-lx")    == 0) lx    = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-ly")    == 0) ly    = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-d")     == 0) { bondd = atoi(argv[i+1]); max_dcut = bondd; }
    else if (strcmp(argv[i], "-sec")   == 0) sec   = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-exci")  == 0) exci  = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-niter") == 0) niter = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-psize") == 0) psize = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-dr")    == 0) read  = atoi(argv[i+1]);
    else if (strcmp(argv[i], "-dge")   == 0) read2 = atoi(argv[i+1]);
  }

  omp_set_num_threads(psize);

  cout << "SU2 DMRG (real): lx=" << lx << " ly=" << ly << " D=" << bondd
       << " 2J=" << sec << " exci=" << exci << " niter=" << niter
       << " threads=" << psize << endl;

  // For SU2: phdim = 2*S + 1 (number of m_J states per site)
  // For spin-1/2: physpn = 1 (2*S=1)
  int physpn = 1;

  DMRG<double, SU2Symmetry> chain(ly, lx, bondd, physpn, sec, exci);
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
    cout<<"done do_idmrg"<<endl;
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
  cout << "E/bond ≈ " << chain.get_enr() / (lx * ly) << endl;

  return 0;
}
