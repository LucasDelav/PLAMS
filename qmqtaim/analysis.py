"""
QTAIM and IQA analysis tools.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scm.plams import KFFile, Molecule

# Pandas is optional - only needed for DataFrame exports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ pandas non disponible - exports CSV/Excel désactivés")


class QTAIMAnalyzer:
    """
    Analyzer for QTAIM and IQA results from ADF calculations.
    """
    
    def __init__(self):
        """Initialize QTAIM analyzer."""
        pass
    
    def analyze_rkf(
        self,
        rkf_path: Path,
        mol: Optional[Molecule] = None,
        topology_analyzer = None,
        conformer_id: Optional[str] = None,
        charge: Optional[int] = None,
        molecule_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract QTAIM and IQA data from RKF file with optional reindexing.

        Args:
            rkf_path: Path to adf.rkf file
            mol: PLAMS Molecule object (for topology analysis)
            topology_analyzer: TopologyAnalyzer instance (for reindexing)
            conformer_id: Conformer identifier (for logging context)
            charge: Charge state (for logging context)
            molecule_name: Molecule name (for logging context)

        Returns:
            Dictionary with analysis results
        """
        # Build context string for logging
        context_parts = []
        if molecule_name:
            context_parts.append(molecule_name)
        if conformer_id:
            context_parts.append(conformer_id)
        if charge is not None:
            context_parts.append(f"charge {charge:+d}" if charge != 0 else "neutral")

        context_str = " ".join(context_parts) if context_parts else rkf_path.name

        print(f"    → Analyzing {context_str}...")
        print(f"      RKF: {rkf_path.name}")

        results = {}
        reindex_map = None

        try:
            kf = KFFile(str(rkf_path))

            # Extract critical points first (needed for reindexing)
            cps = self._extract_critical_points(kf)
            if cps:
                results['critical_points'] = cps
                print(f"      ✓ Found {len(cps)} ring critical point(s)")

            # Attempt reindexing if mol and topology_analyzer are provided
            if mol is not None and topology_analyzer is not None and cps:
                try:
                    # Select best ring CP using geometric validation
                    selected_cp = topology_analyzer.find_ring_critical_point(
                        cps,
                        ring_size=6,
                        mol=mol,
                        expected_oxygens=2
                    )

                    cp_coords = np.array(selected_cp['coords'])
                    reverse_map = topology_analyzer.reindex_atoms(mol, cp_coords)

                    # Create forward map: original_index -> canonical_index
                    reindex_map = {v: k for k, v in reverse_map.items()}

                    results['reindexed'] = True
                    results['reindex_map'] = reindex_map
                    print(f"      ✓ Topology reindexing successful")
                    print(f"        Mapping (AMS→Canonical): {reindex_map}")

                except ValueError as e:
                    # Strict validation failed (0 or 2+ valid rCPs)
                    # This is an expected failure mode - skip this conformer/charge
                    print(f"      ✗ Ring CP validation failed: {e}")
                    print(f"        Skipping this conformer/charge state")
                    return {'error': 'ring_cp_validation_failed', 'message': str(e)}

                except Exception as e:
                    # Unexpected error during reindexing
                    import traceback
                    print(f"      ✗ Unexpected reindexing error: {e}")
                    print(f"        Traceback:")
                    for line in traceback.format_exc().split('\n'):
                        if line.strip():
                            print(f"          {line}")
                    print(f"        Skipping this conformer/charge state")
                    return {'error': 'reindexing_error', 'message': str(e)}
            else:
                results['reindexed'] = False

            # Extract Bader charges
            bader_charges = self._extract_bader_charges(kf)
            if bader_charges:
                # Apply reindexing if available
                if reindex_map:
                    bader_charges = self._apply_reindex(bader_charges, reindex_map)
                results['bader_charges'] = bader_charges
                print(f"      ✓ Extracted {len(bader_charges)} Bader charges")

            # Extract IQA energies
            iqa_data = self._extract_iqa_energies(kf)
            if iqa_data:
                # Apply reindexing if available
                if reindex_map:
                    if 'iqa_Eintra' in iqa_data:
                        iqa_data['iqa_Eintra'] = self._apply_reindex(
                            iqa_data['iqa_Eintra'], reindex_map
                        )
                    if 'iqa_EaddIQA' in iqa_data:
                        iqa_data['iqa_EaddIQA'] = self._apply_reindex(
                            iqa_data['iqa_EaddIQA'], reindex_map
                        )
                results.update(iqa_data)
                if 'iqa_total' in iqa_data:
                    print(f"      ✓ Extracted IQA energies (total: {iqa_data['iqa_total']:.6f} Ha)")

            return results

        except Exception as e:
            print(f"      ✗ Error analyzing RKF: {str(e)}")
            return {}

    def _apply_reindex(
        self,
        data_dict: Dict[int, Any],
        reindex_map: Dict[int, int]
    ) -> Dict[int, Any]:
        """
        Apply reindexing to a dictionary of atom indices.

        Args:
            data_dict: Dictionary with input atom indices as keys
            reindex_map: Mapping from input index -> canonical index

        Returns:
            Dictionary with canonical indices (1-8) for ring atoms,
            and input indices for other atoms (H, halogens, etc.)

        Note:
            Non-mapped atoms are processed FIRST so that mapped atoms
            can overwrite in case of index collision (e.g., input idx 1
            might collide with canonical idx 1 for C1).
        """
        reindexed = {}

        # First pass: add non-mapped atoms (may be overwritten later)
        for input_idx, value in data_dict.items():
            if input_idx not in reindex_map:
                reindexed[input_idx] = value

        # Second pass: add mapped atoms (overwrite if collision)
        for input_idx, value in data_dict.items():
            canonical_idx = reindex_map.get(input_idx)
            if canonical_idx is not None:
                reindexed[canonical_idx] = value

        return reindexed

    def _get_atom_order_mapping(self, kf: KFFile) -> List[int]:
        """
        Get mapping from internal AMS order to input order.

        AMS reorders atoms internally (typically by atom type).
        The 'atom order index' array has shape [n_atoms, 2]:
          - First n_atoms elements (col 0): input_idx → internal_idx
          - Second n_atoms elements (col 1): internal_idx → input_idx

        We need the SECOND column (internal→input) to convert arrays
        stored in internal order (like Bader charges) back to input order.

        Args:
            kf: KFFile object

        Returns:
            List where index i gives the input index for internal index i
        """
        try:
            atom_order = kf.read("Geometry", "atom order index")
            # Array has 2*n_atoms elements, get n_atoms from array length
            n_atoms = len(atom_order) // 2
            # Second half of array is internal→input mapping
            # atom_order[n_atoms + internal_idx] = input_idx
            return [int(x) for x in atom_order[n_atoms:2*n_atoms]]
        except KeyError:
            # Fallback: cannot determine mapping without atom order index
            return []

    def _extract_bader_charges(self, kf: KFFile) -> Optional[Dict[int, float]]:
        """
        Extract Bader charges from Properties section.

        IMPORTANT: Les charges sont dans l'ordre interne AMS, pas l'ordre d'entrée.
        On utilise 'atom order index' pour convertir vers l'ordre d'entrée.
        """
        try:
            charges = kf.read("Properties", "Bader atomic charges")

            # Get mapping: internal_idx -> input_idx
            atom_order = self._get_atom_order_mapping(kf)

            # Map charges from internal order to input order
            charges_input_order = {}
            for internal_idx, charge in enumerate(charges):
                if internal_idx < len(atom_order):
                    input_idx = atom_order[internal_idx]
                    charges_input_order[input_idx] = float(charge)
                else:
                    # Fallback: use 1-indexed if no mapping
                    charges_input_order[internal_idx + 1] = float(charge)

            return charges_input_order

        except KeyError:
            print("      ⚠ Bader charges not found in RKF")
            return None
    
    def _extract_iqa_energies(self, kf: KFFile) -> Dict[str, Any]:
        """
        Extract IQA additive energies (EaddIQA) from Properties section.

        The "IQA atom-atom total" array contains:
        - Diagonal elements: Eintra(A) - intra-atomic energies
        - Off-diagonal elements: Einter(A,B) - interaction energies

        Storage format: Upper triangular, row-major order, INPUT atom order.

        Formula: EaddIQA(A) = Eintra(A) + ½ × Σ_{B≠A} Einter(A,B)

        See docs/IQA_DATA_STRUCTURE.md for detailed documentation.
        """
        results = {}

        try:
            # Read the IQA atom-atom total array
            atom_atom = kf.read("Properties", "IQA atom-atom total")
            n_atoms = int(kf.read("Geometry", "nr of atoms"))

            # Verify array size (should be n*(n+1)/2 for triangular matrix)
            expected_size = n_atoms * (n_atoms + 1) // 2
            if len(atom_atom) != expected_size:
                print(f"      ⚠ IQA array size mismatch: got {len(atom_atom)}, expected {expected_size}")
                return {}

            # Convert upper triangular row-major to full symmetric matrix
            matrix = self._tri_to_full_matrix(atom_atom, n_atoms)

            # Extract Eintra (diagonal) and compute EaddIQA
            Eintra = {}
            EaddIQA = {}

            for i in range(n_atoms):
                Eintra[i + 1] = float(matrix[i, i])  # 1-indexed

                # EaddIQA = Eintra + ½ × Σ_B Einter(A,B)
                Einter_sum = sum(matrix[i, j] for j in range(n_atoms) if j != i)
                EaddIQA[i + 1] = float(matrix[i, i] + 0.5 * Einter_sum)

            results['iqa_Eintra'] = Eintra
            results['iqa_EaddIQA'] = EaddIQA
            results['iqa_total'] = sum(EaddIQA.values())

            # Also store the full interaction matrix for detailed analysis
            # Convert to dict format: {"i-j": Einter} for i < j (string keys for JSON compatibility)
            Einter_pairs = {}
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    key = f"{i + 1}-{j + 1}"
                    Einter_pairs[key] = float(matrix[i, j])
            results['iqa_Einter'] = Einter_pairs

            return results

        except KeyError as e:
            print(f"      ⚠ IQA data not found: {str(e)}")
            return {}

    def _tri_to_full_matrix(self, tri_array: list, n: int) -> np.ndarray:
        """
        Convert upper triangular row-major array to full symmetric matrix.

        Storage order: (0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ..., (n-1,n-1)

        Args:
            tri_array: Flat array of n(n+1)/2 elements
            n: Matrix dimension

        Returns:
            Full symmetric n×n matrix
        """
        matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = tri_array[idx]
                matrix[j, i] = tri_array[idx]  # Symmetric
                idx += 1
        return matrix
    
    def _extract_critical_points(self, kf: KFFile) -> List[Dict]:
        """
        Extract critical points from QTAIM, especially ring CPs (code=4).

        Ring CP (code=4) coordinates are extracted from the CP coordinates array
        which is organized as [x1...xN, y1...yN, z1...zN] where N is the total
        number of critical points.

        Also extracts CP densities to select the best ring CP when multiple are found.
        """
        try:
            # 1. Lire les codes des CP (rank, signature)
            cp_codes = kf.read("Properties", "CP code number for (Rank,Signatu")

            # 2. Trouver les positions des ring CP (code = 4)
            ring_cp_indices = [i for i, code in enumerate(cp_codes) if code == 4]

            if not ring_cp_indices:
                print("      ⚠ No ring critical points (code=4) found")
                return []

            # 3. Lire le nombre total de CP
            n_cp = int(kf.read("Properties", "CP number of"))

            # 4. Lire les coordonnées (format: x1...xN, y1...yN, z1...zN)
            cp_coords = kf.read("Properties", "CP coordinates")

            # 5. Lire les densités aux points critiques
            try:
                cp_densities = kf.read("Properties", "CP density at")
            except KeyError:
                print("      ⚠ CP densities not found, will use first ring CP")
                cp_densities = None

            # 6. Extraire les coordonnées et densités des ring CP
            ring_cps = []
            for idx in ring_cp_indices:
                cp = {
                    'type': 4,  # Ring CP
                    'index': idx,
                    'coords': [
                        float(cp_coords[idx]),           # x
                        float(cp_coords[idx + n_cp]),    # y
                        float(cp_coords[idx + 2*n_cp])   # z
                    ]
                }

                # Ajouter la densité si disponible
                if cp_densities is not None:
                    cp['density'] = float(cp_densities[idx])

                ring_cps.append(cp)

            print(f"      ✓ Found {len(ring_cps)} ring critical point(s)")

            # Afficher les densités si multiples rCP trouvés
            if len(ring_cps) > 1 and cp_densities is not None:
                print(f"      → Ring CP densities:")
                for i, cp in enumerate(ring_cps):
                    print(f"         rCP {i+1} (index {cp['index']}): ρ = {cp['density']:.6f}")

            return ring_cps

        except KeyError as e:
            print(f"      ⚠ Critical points not found: {e}")
            return []
    
    def export_results(
        self,
        results: Dict[str, Dict[int, Any]],
        output_dir: Path,
        workdir: Path = None
    ) -> None:
        """
        Export results to CSV files.

        Args:
            results: Results dictionary {conformer_id: {charge: data}}
            output_dir: Output directory for CSV files
            workdir: Molecule workdir path (optional, for Boltzmann analysis)
        """
        print("\nExporting results...")

        # Export Bader charges for each charge state
        for charge in [-1, 0, +1]:
            self._export_bader_charges_by_charge(results, output_dir, charge)

        # Export IQA energies for each charge state
        for charge in [-1, 0, +1]:
            self._export_iqa_energies(results, output_dir, charge)

        # Export Boltzmann-weighted analysis if workdir provided
        if workdir is not None:
            for charge in [-1, 0, +1]:
                self._export_boltzmann_analysis(results, output_dir, charge, workdir)

            # Export text summary
            self._export_text_summary(results, output_dir, workdir)

        # Export complete data to JSON
        self._export_json(results, output_dir)

        print("✓ Results exported")
    
    def _export_bader_charges_by_charge(
        self,
        results: Dict[str, Dict[int, Any]],
        output_dir: Path,
        charge: int
    ) -> None:
        """Export Bader charges for a specific charge state."""
        rows = []
        
        for conf_id, charge_data in results.items():
            if charge not in charge_data:
                continue
            
            data = charge_data[charge]
            if 'bader_charges' not in data:
                continue
            
            row = {'conformer_id': conf_id}
            
            # Add charges for each atom (assuming indices 1-8)
            for idx in range(1, 9):
                if idx in data['bader_charges']:
                    label = f"C{idx}" if idx <= 6 else f"O{idx}"
                    row[label] = data['bader_charges'][idx]
            
            rows.append(row)
        
        if rows:
            if not HAS_PANDAS:
                print("  ⚠ Pandas non disponible - export CSV ignoré")
                return

            df = pd.DataFrame(rows)
            charge_label = f"charge{charge:+d}" if charge != 0 else "neutral"
            filepath = output_dir / f"bader_charges_{charge_label}.csv"
            df.to_csv(filepath, index=False)
            print(f"  → {filepath.name}")
    
    def _export_iqa_energies(
        self,
        results: Dict[str, Dict[int, Any]],
        output_dir: Path,
        charge: int
    ) -> None:
        """Export IQA additive energies (EaddIQA) for a specific charge state."""
        rows = []

        for conf_id, charge_data in results.items():
            if charge not in charge_data:
                continue

            data = charge_data[charge]
            if 'iqa_EaddIQA' not in data:
                continue

            row = {'conformer_id': conf_id}

            # IQA additive energies and intra-atomic energies
            for idx in range(1, 19):  # Support up to 18 atoms
                if 'iqa_EaddIQA' in data and idx in data['iqa_EaddIQA']:
                    row[f"EaddIQA_{idx}"] = data['iqa_EaddIQA'][idx]
                if 'iqa_Eintra' in data and idx in data['iqa_Eintra']:
                    row[f"Eintra_{idx}"] = data['iqa_Eintra'][idx]

            # Total energy
            if 'iqa_total' in data:
                row['total'] = data['iqa_total']

            rows.append(row)

        if rows:
            if not HAS_PANDAS:
                print("  ⚠ Pandas non disponible - export CSV ignoré")
                return

            df = pd.DataFrame(rows)
            charge_label = f"charge{charge:+d}" if charge != 0 else "neutral"
            filepath = output_dir / f"iqa_energies_{charge_label}.csv"
            df.to_csv(filepath, index=False)
            print(f"  → {filepath.name}")

    def _read_boltzmann_weights(self, workdir: Path) -> Optional[Dict[str, float]]:
        """
        Read Boltzmann weights from conformers summary file.

        Args:
            workdir: Molecule workdir path

        Returns:
            Dictionary {conformer_id: weight} or None if file not found
        """
        summary_file = workdir / "conformers" / "results" / "conformers_summary.txt"

        if not summary_file.exists():
            print(f"  ⚠ Conformers summary not found: {summary_file}")
            return None

        weights = {}
        try:
            with open(summary_file, 'r') as f:
                for line in f:
                    # Format: "conf_1: Energy = ... Boltzmann weight = 0.xxxx"
                    if "Boltzmann weight" in line:
                        parts = line.split()
                        conf_id = parts[0].rstrip(':')  # Remove trailing ':'
                        weight_idx = parts.index('weight') + 2  # Skip '=' sign
                        weight = float(parts[weight_idx])
                        weights[conf_id] = weight
        except Exception as e:
            print(f"  ⚠ Error reading Boltzmann weights: {e}")
            return None

        return weights

    def _export_boltzmann_analysis(
        self,
        results: Dict[str, Dict[int, Any]],
        output_dir: Path,
        charge: int,
        workdir: Path
    ) -> None:
        """
        Export Boltzmann-weighted averages and standard deviations.

        Args:
            results: Results dictionary {conformer_id: {charge: data}}
            output_dir: Output directory for CSV files
            charge: Charge state to analyze
            workdir: Molecule workdir path (to read Boltzmann weights)
        """
        # Read Boltzmann weights
        weights = self._read_boltzmann_weights(workdir)
        if weights is None:
            print(f"  ⚠ Skipping Boltzmann analysis for charge {charge} (no weights available)")
            return

        # Collect data for each atom
        bader_data = {idx: [] for idx in range(1, 9)}  # C1-C6, O7-O8
        iqa_data = {idx: [] for idx in range(1, 9)}
        conf_weights = []

        for conf_id, charge_data in results.items():
            if charge not in charge_data:
                continue

            if conf_id not in weights:
                print(f"  ⚠ No Boltzmann weight for {conf_id}, skipping")
                continue

            data = charge_data[charge]
            weight = weights[conf_id]
            conf_weights.append(weight)

            # Collect Bader charges
            if 'bader_charges' in data:
                for idx in range(1, 9):
                    if idx in data['bader_charges']:
                        bader_data[idx].append((data['bader_charges'][idx], weight))

            # Collect IQA additive energies
            if 'iqa_EaddIQA' in data:
                for idx in range(1, 9):
                    if idx in data['iqa_EaddIQA']:
                        iqa_data[idx].append((data['iqa_EaddIQA'][idx], weight))

        if not conf_weights:
            print(f"  ⚠ No data for Boltzmann analysis (charge {charge})")
            return

        # Normalize weights (should sum to 1.0, but just in case)
        total_weight = sum(conf_weights)

        # Calculate Boltzmann-weighted averages and std devs
        rows = []

        for idx in range(1, 9):
            label = f"C{idx}" if idx <= 6 else f"O{idx}"
            row = {'atom': label}

            # Bader charge statistics
            if bader_data[idx]:
                values = np.array([v for v, w in bader_data[idx]])
                weights_arr = np.array([w for v, w in bader_data[idx]]) / total_weight

                mean_bader = np.sum(values * weights_arr)
                var_bader = np.sum(weights_arr * (values - mean_bader)**2)
                std_bader = np.sqrt(var_bader)

                row['bader_mean'] = mean_bader
                row['bader_std'] = std_bader

            # IQA energy statistics
            if iqa_data[idx]:
                values = np.array([v for v, w in iqa_data[idx]])
                weights_arr = np.array([w for v, w in iqa_data[idx]]) / total_weight

                mean_iqa = np.sum(values * weights_arr)
                var_iqa = np.sum(weights_arr * (values - mean_iqa)**2)
                std_iqa = np.sqrt(var_iqa)

                row['iqa_mean'] = mean_iqa
                row['iqa_std'] = std_iqa

            rows.append(row)

        # Export to CSV
        if rows:
            if not HAS_PANDAS:
                print("  ⚠ Pandas non disponible - export Boltzmann analysis ignoré")
                return

            df = pd.DataFrame(rows)
            charge_label = f"charge{charge:+d}" if charge != 0 else "neutral"
            filepath = output_dir / f"boltzmann_analysis_{charge_label}.csv"
            df.to_csv(filepath, index=False)
            print(f"  → {filepath.name}")

    def _export_text_summary(
        self,
        results: Dict[str, Dict[int, Any]],
        output_dir: Path,
        workdir: Path
    ) -> None:
        """
        Export a comprehensive text summary of QTAIM/IQA results.

        Args:
            results: Results dictionary {conformer_id: {charge: data}}
            output_dir: Output directory
            workdir: Molecule workdir path
        """
        filepath = output_dir / "qtaim_summary.txt"

        # Read Boltzmann weights
        weights = self._read_boltzmann_weights(workdir)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("QTAIM/IQA ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Conformers section
            f.write(f"Number of conformers analyzed: {len(results)}\n\n")

            if weights:
                f.write("Boltzmann Weights:\n")
                f.write("-" * 40 + "\n")
                for conf_id in sorted(results.keys()):
                    if conf_id in weights:
                        f.write(f"  {conf_id}: {weights[conf_id]:.4f}\n")
                f.write("\n")

            # Analysis for each charge state
            for charge in [-1, 0, +1]:
                charge_label = f"Charge {charge:+d}" if charge != 0 else "Neutral"
                f.write("=" * 80 + "\n")
                f.write(f"{charge_label} State\n")
                f.write("=" * 80 + "\n\n")

                # Collect data for averaging
                bader_data = {idx: [] for idx in range(1, 9)}
                iqa_data = {idx: [] for idx in range(1, 9)}

                for conf_id, charge_data in results.items():
                    if charge not in charge_data:
                        continue

                    data = charge_data[charge]
                    weight = weights.get(conf_id, 1.0 / len(results)) if weights else 1.0 / len(results)

                    if 'bader_charges' in data:
                        for idx in range(1, 9):
                            if idx in data['bader_charges']:
                                bader_data[idx].append((data['bader_charges'][idx], weight))

                    if 'iqa_EaddIQA' in data:
                        for idx in range(1, 9):
                            if idx in data['iqa_EaddIQA']:
                                iqa_data[idx].append((data['iqa_EaddIQA'][idx], weight))

                # Calculate and write Bader charges
                f.write("Bader Atomic Charges (Boltzmann-weighted):\n")
                f.write("-" * 40 + "\n")
                for idx in range(1, 9):
                    label = f"C{idx}" if idx <= 6 else f"O{idx}"
                    if bader_data[idx]:
                        values = np.array([v for v, w in bader_data[idx]])
                        weights_arr = np.array([w for v, w in bader_data[idx]])
                        weights_arr = weights_arr / weights_arr.sum()

                        mean = np.sum(values * weights_arr)
                        var = np.sum(weights_arr * (values - mean)**2)
                        std = np.sqrt(var)

                        f.write(f"  {label}: {mean:+.4f} ± {std:.4f} e\n")
                f.write("\n")

                # Calculate and write IQA additive energies
                f.write("IQA Additive Energies EaddIQA (Boltzmann-weighted, Hartree):\n")
                f.write("-" * 40 + "\n")
                for idx in range(1, 9):
                    label = f"C{idx}" if idx <= 6 else f"O{idx}"
                    if iqa_data[idx]:
                        values = np.array([v for v, w in iqa_data[idx]])
                        weights_arr = np.array([w for v, w in iqa_data[idx]])
                        weights_arr = weights_arr / weights_arr.sum()

                        mean = np.sum(values * weights_arr)
                        var = np.sum(weights_arr * (values - mean)**2)
                        std = np.sqrt(var)

                        f.write(f"  {label}: {mean:.6f} ± {std:.6f} Eh\n")
                f.write("\n")

            # Comparison section (if multiple charge states available)
            f.write("=" * 80 + "\n")
            f.write("Charge State Comparison\n")
            f.write("=" * 80 + "\n\n")

            charge_states = [-1, 0, +1]
            charge_labels = ["Reduced", "Neutral", "Oxidized"]

            # Compare Bader charges for key atoms
            f.write("Bader Charge Differences (Boltzmann-weighted):\n")
            f.write("-" * 40 + "\n")

            # Calculate averages for each charge state
            avg_bader = {}
            for charge in charge_states:
                avg_bader[charge] = {}
                for conf_id, charge_data in results.items():
                    if charge not in charge_data:
                        continue
                    data = charge_data[charge]
                    weight = weights.get(conf_id, 1.0 / len(results)) if weights else 1.0 / len(results)

                    if 'bader_charges' in data:
                        for idx in range(1, 9):
                            if idx in data['bader_charges']:
                                if idx not in avg_bader[charge]:
                                    avg_bader[charge][idx] = []
                                avg_bader[charge][idx].append((data['bader_charges'][idx], weight))

            # Compute weighted means
            for charge in charge_states:
                for idx in avg_bader[charge]:
                    values = np.array([v for v, w in avg_bader[charge][idx]])
                    weights_arr = np.array([w for v, w in avg_bader[charge][idx]])
                    weights_arr = weights_arr / weights_arr.sum()
                    avg_bader[charge][idx] = np.sum(values * weights_arr)

            # Write comparison
            for idx in range(1, 9):
                label = f"C{idx}" if idx <= 6 else f"O{idx}"
                if all(idx in avg_bader[c] for c in charge_states):
                    f.write(f"  {label}:\n")
                    for i, (charge, charge_label) in enumerate(zip(charge_states, charge_labels)):
                        f.write(f"    {charge_label:10s}: {avg_bader[charge][idx]:+.4f} e\n")
                    delta_red = avg_bader[-1][idx] - avg_bader[0][idx]
                    delta_ox = avg_bader[+1][idx] - avg_bader[0][idx]
                    f.write(f"    Δ(Red-Neu): {delta_red:+.4f} e\n")
                    f.write(f"    Δ(Ox-Neu):  {delta_ox:+.4f} e\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("End of Summary\n")
            f.write("=" * 80 + "\n")

        print(f"  → {filepath.name}")

    def _export_json(
        self,
        results: Dict[str, Dict[int, Any]],
        output_dir: Path
    ) -> None:
        """
        Export complete results data to JSON format.

        Args:
            results: Results dictionary {conformer_id: {charge: data}}
            output_dir: Output directory
        """
        import json

        filepath = output_dir / "qtaim_complete_data.json"

        # Convert numpy arrays and other non-serializable types
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        # Convert results to serializable format
        serializable_results = convert_to_serializable(results)

        # Write JSON with pretty formatting
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"  → {filepath.name}")

