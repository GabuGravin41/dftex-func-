"""
One-Electron Diagnostics for Machine-Learned Exchange-Correlation Functionals
Comprehensive Verification & Stability Suite
Author: Dalton Omondi / AutoBot Research Suite
"""

import sys
import subprocess
import os

print("=" * 80)
print("ONE-ELECTRON DIAGNOSTICS FOR ML-XC FUNCTIONALS - VERIFICATION RUNNER")
print("=" * 80)

# Step 1: Install prerequisites
print("\n[STEP 1] Checking and installing dependencies...")
packages = ["pyscf", "ciderpress", "pandas", "tabulate"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg} is already installed.")
    except ImportError:
        print(f"  [INSTALL] Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

import numpy as np
import pandas as pd
from tabulate import tabulate
import pyscf
from pyscf import gto, scf, dft
from pyscf.dft import rks, uks

print(f"\nPySCF version: {pyscf.__version__}")
try:
    import ciderpress
    print(f"CiderPress loaded successfully: {ciderpress.__file__}")
    cider_available = True
except Exception as e:
    print(f"[WARN] CiderPress import error: {e}")
    cider_available = False

master_records = []

# =====================================================================
# PART 1: ISOELECTRONIC SERIES & BASIS-INCOMPLETENESS (TABLE 4.1 & 4.3)
# =====================================================================
print("\n" + "=" * 80)
print("[PART 1] Isoelectronic Series & Basis Incompleteness (def2-TZVP, def2-SVP, def2-QZVP, aug-cc-pVTZ)")
print("=" * 80)

ions = [
    ("H", 1, 0, 1),
    ("He+", 2, 1, 1),
    ("Li2+", 3, 2, 1),
    ("Be3+", 4, 3, 1),
]

basis_list = ["def2-tzvp", "def2-svp", "def2-qzvp", "aug-cc-pvtz"]
basis_incomp_results = []

for basis in basis_list:
    print(f"\n--- Testing Basis: {basis} ---")
    for symbol, z, charge, spin in ions:
        e_exact = -0.5 * (z ** 2)
        
        # Build molecule
        mol = gto.M(
            atom=f"{symbol.rstrip('0123456789+-')} 0 0 0",
            basis=basis,
            charge=charge,
            spin=spin,
            verbose=0
        )
        
        # 1. Exact Same-Basis UHF
        mf_uhf = scf.UHF(mol)
        mf_uhf.conv_tol = 1e-12
        mf_uhf.max_cycle = 200
        mf_uhf.kernel()
        e_uhf = mf_uhf.e_tot
        incomp_mha = (e_uhf - e_exact) * 1000.0
        
        basis_incomp_results.append({
            "Basis": basis,
            "Ion": symbol,
            "Z": z,
            "E_exact (Ha)": e_exact,
            "E_UHF (Ha)": e_uhf,
            "Incompleteness (mHa)": incomp_mha,
            "Converged": mf_uhf.converged,
            "Cycles": mf_uhf.cycles
        })
        
        master_records.append({
            "functional": "UHF",
            "system": symbol,
            "basis": basis,
            "R": None,
            "E": e_uhf,
            "converged": mf_uhf.converged,
            "cycles": mf_uhf.cycles,
            "stable": True,
            "q_Ha": 1.0 - charge,
            "q_Hb": None,
            "notes": "Exact reference in basis"
        })

df_basis = pd.DataFrame(basis_incomp_results)
print("\n" + tabulate(df_basis, headers="keys", tablefmt="github", floatfmt=".6f"))

# =====================================================================
# PART 2: ISOELECTRONIC SERIES FOR DFT FUNCTIONALS (TABLE 4.2 & 4.3)
# =====================================================================
print("\n" + "=" * 80)
print("[PART 2] Isoelectronic Series DFT SIE: PBE, r2SCAN, SCAN, B3LYP (def2-TZVP)")
print("=" * 80)

dft_functionals = ["PBE", "r2SCAN", "SCAN", "B3LYP"]
primary_basis = "def2-tzvp"
iso_dft_results = []

for xc in dft_functionals:
    print(f"\nRunning Functional: {xc}...")
    for symbol, z, charge, spin in ions:
        e_exact = -0.5 * (z ** 2)
        mol = gto.M(
            atom=f"{symbol.rstrip('0123456789+-')} 0 0 0",
            basis=primary_basis,
            charge=charge,
            spin=spin,
            verbose=0
        )
        
        # Reference UHF in same basis
        mf_uhf = scf.UHF(mol)
        mf_uhf.conv_tol = 1e-12
        mf_uhf.kernel()
        e_uhf = mf_uhf.e_tot
        
        # DFT calculation
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.conv_tol = 1e-11
        mf.max_cycle = 200
        mf.kernel()
        e_dft = mf.e_tot
        
        # Check stability
        mo_stable, is_stable = mf.stability(verbose=0)
        
        naive_sie_mha = (e_dft - e_exact) * 1000.0
        same_sie_mha = (e_dft - e_uhf) * 1000.0
        ratio = naive_sie_mha / same_sie_mha if abs(same_sie_mha) > 1e-6 else np.nan
        
        iso_dft_results.append({
            "Functional": xc,
            "Ion": symbol,
            "Z": z,
            "E_DFT (Ha)": e_dft,
            "E_UHF (Ha)": e_uhf,
            "Naive SIE (mHa)": naive_sie_mha,
            "Same-Basis SIE (mHa)": same_sie_mha,
            "Inflation Ratio": ratio,
            "Converged": mf.converged,
            "Stable": is_stable
        })
        
        master_records.append({
            "functional": xc,
            "system": symbol,
            "basis": primary_basis,
            "R": None,
            "E": e_dft,
            "converged": mf.converged,
            "cycles": mf.cycles,
            "stable": is_stable,
            "q_Ha": 1.0 - charge,
            "q_Hb": None,
            "notes": f"Naive SIE: {naive_sie_mha:.3f} mHa, Same SIE: {same_sie_mha:.3f} mHa"
        })

df_iso_dft = pd.DataFrame(iso_dft_results)
print("\n" + tabulate(df_iso_dft, headers="keys", tablefmt="github", floatfmt=".4f"))

# =====================================================================
# PART 3: H2+ DISSOCIATION CURVE & B3LYP ANOMALY DIAGNOSIS (TABLE 4.4)
# =====================================================================
print("\n" + "=" * 80)
print("[PART 3] H2+ Dissociation Curve, Stability Analysis, and B3LYP Anomaly Diagnosis")
print("=" * 80)

r_grid = [0.5, 0.75, 1.06, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 6.0, 7.0, 10.0, 15.0]
test_functionals = ["PBE", "r2SCAN", "B3LYP"]

h2plus_results = []

for xc in test_functionals:
    print(f"\n>>> SCANNING H2+ WITH FUNCTIONAL: {xc} <<<")
    for r in r_grid:
        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {r}",
            basis=primary_basis,
            charge=1,
            spin=1,
            verbose=0
        )
        
        # 1. Exact Same-Basis UHF Reference
        mf_uhf = scf.UHF(mol)
        mf_uhf.conv_tol = 1e-12
        mf_uhf.kernel()
        e_uhf = mf_uhf.e_tot
        uhf_charges = mf_uhf.mulliken_pop(verbose=0)[1]
        
        # 2. Standard UKS with default guess
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.conv_tol = 1e-10
        mf.max_cycle = 200
        mf.kernel()
        e_dft = mf.e_tot
        dft_charges = mf.mulliken_pop(verbose=0)[1]
        
        # Stability check
        try:
            mo_stable, is_stable = mf.stability(verbose=0)
        except Exception as e:
            mo_stable, is_stable = None, False
        
        # If unstable or at stretched geometry, test broken-symmetry / localized initial guess
        e_broken = None
        broken_charges = None
        broken_converged = None
        broken_stable = None
        
        if r >= 2.5:
            # Force broken-symmetry guess: localize density onto atom 1
            mf_bs = dft.UKS(mol)
            mf_bs.xc = xc
            mf_bs.conv_tol = 1e-10
            mf_bs.max_cycle = 200
            
            # Construct localized density guess
            dm_init = mf.make_rdm1()
            # Perturb alpha spin density towards atom 0
            dm_init[0][:, :] *= 1.5
            dm_init[0][mol.search_ao_label('0 H'), :] *= 2.0
            
            mf_bs.kernel(dm0=dm_init)
            e_broken = mf_bs.e_tot
            broken_charges = mf_bs.mulliken_pop(verbose=0)[1]
            broken_converged = mf_bs.converged
            try:
                _, broken_stable = mf_bs.stability(verbose=0)
            except:
                broken_stable = False
        
        delta_mha = (e_dft - e_uhf) * 1000.0
        
        res_entry = {
            "Functional": xc,
            "R (A)": r,
            "E_DFT (Ha)": e_dft,
            "E_UHF (Ha)": e_uhf,
            "Delta (mHa)": delta_mha,
            "qa": dft_charges[0],
            "qb": dft_charges[1],
            "Converged": mf.converged,
            "Cycles": mf.cycles,
            "Stable": is_stable,
            "E_broken (Ha)": e_broken if e_broken is not None else np.nan,
            "qa_broken": broken_charges[0] if broken_charges is not None else np.nan,
            "qb_broken": broken_charges[1] if broken_charges is not None else np.nan,
            "Broken_Stable": broken_stable if broken_stable is not None else False
        }
        h2plus_results.append(res_entry)
        
        master_records.append({
            "functional": xc,
            "system": "H2+",
            "basis": primary_basis,
            "R": r,
            "E": e_dft,
            "converged": mf.converged,
            "cycles": mf.cycles,
            "stable": is_stable,
            "q_Ha": dft_charges[0],
            "q_Hb": dft_charges[1],
            "notes": f"Delta: {delta_mha:.3f} mHa. Broken E: {e_broken}"
        })
        
        print(f"  R={r:5.2f} A | E_DFT: {e_dft:.6f} Ha | E_UHF: {e_uhf:.6f} Ha | Delta: {delta_mha:+8.3f} mHa | qa={dft_charges[0]:.3f}, qb={dft_charges[1]:.3f} | Stable: {is_stable} | Conv: {mf.converged}")
        if e_broken is not None and abs(e_broken - e_dft) > 1e-5:
            delta_bs = (e_broken - e_uhf) * 1000.0
            print(f"      [BROKEN BRANCH FOUND] E_BS: {e_broken:.6f} Ha | Delta_BS: {delta_bs:+8.3f} mHa | qa_BS={broken_charges[0]:.3f}, qb_BS={broken_charges[1]:.3f} | Stable: {broken_stable}")

df_h2plus = pd.DataFrame(h2plus_results)

# =====================================================================
# PART 4: CIDER INVESTIGATION (IF CIDERPRESS AVAILABLE)
# =====================================================================
if cider_available:
    print("\n" + "=" * 80)
    print("[PART 4] CIDER Functional Investigation on H2+ Curve")
    print("=" * 80)
    try:
        # Check what CIDER models exist
        import inspect
        print("CiderPress modules:", dir(ciderpress))
        
        # Test CIDER registration on PySCF
        # We will attempt to run CIDER across R in [0.5, 1.06, 2.0, 3.0, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 7.0, 10.0]
        # and record stability & convergence
    except Exception as e:
        print(f"Error initializing CIDER test: {e}")

# =====================================================================
# PART 5: SAVE MASTER CSV
# =====================================================================
print("\n" + "=" * 80)
print("[PART 5] Exporting Canonical Master CSV")
print("=" * 80)

df_master = pd.DataFrame(master_records)
csv_path = "master_dft_one_electron_dataset.csv"
df_master.to_csv(csv_path, index=False)
print(f"Saved {len(df_master)} master records to {csv_path}")

print("\n" + "=" * 80)
print("VERIFICATION SUITE COMPLETE")
print("=" * 80)
