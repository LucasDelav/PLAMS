"""
Topological analysis for atom reindexing based on molecular structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scm.plams import Molecule
import logging


class TopologyAnalyzer:
    """
    Analyzer for molecular topology and atom reindexing.
    Identifies the 6-membered ring and assigns canonical indices.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, verbose: bool = False):
        """
        Initialize topology analyzer.

        Args:
            logger: Logger instance
            verbose: Enable verbose diagnostic output (default: False)
        """
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO
            # Add console handler if none exists
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

        self.logger = logger
        self.verbose = verbose
        self.ring_atoms = []  # Indices of 6-membered ring atoms
        self.oxygen_atoms = []  # Indices of oxygen atoms
        self.reindex_map = {}  # Map: original_index -> canonical_index
        self.reverse_map = {}  # Map: canonical_index -> original_index

    def _print_molecular_structure(self, mol: Molecule) -> None:
        """Print complete molecular structure for debugging."""
        print("="*60)
        print("MOLECULAR STRUCTURE DIAGNOSTIC")
        print("="*60)
        print(f"Total atoms: {len(mol.atoms)}")
        print("")
        print("Atom listing (index, symbol, coords):")
        for i, atom in enumerate(mol.atoms, start=1):
            print(f"  [{i:2d}] {atom.symbol:2s} : ({atom.x:8.4f}, {atom.y:8.4f}, {atom.z:8.4f})")
        print("")

    def _print_connectivity_graph(self, mol: Molecule) -> None:
        """Print complete connectivity graph for debugging."""
        print("CONNECTIVITY GRAPH")
        print("-"*60)
        for i, atom in enumerate(mol.atoms, start=1):
            neighbors = []
            for bond in atom.bonds:
                other = bond.other_end(atom)
                other_idx = mol.atoms.index(other) + 1
                neighbors.append(f"{other_idx}({other.symbol})")

            neighbor_str = ", ".join(neighbors) if neighbors else "none"
            print(f"  Atom {i:2d} ({atom.symbol}): {len(atom.bonds)} bonds → [{neighbor_str}]")
        print("")

    def _print_ring_connectivity(self, mol: Molecule, ring_atoms: List[int]) -> None:
        """Print connectivity matrix for ring atoms."""
        print("RING CONNECTIVITY MATRIX")
        print("-"*60)
        ring_set = set(ring_atoms)

        # Build adjacency matrix
        print("     " + " ".join(f"{idx:3d}" for idx in ring_atoms))
        for i in ring_atoms:
            row = f"{i:3d}:"
            for j in ring_atoms:
                if i == j:
                    row += "  - "
                else:
                    # Check if i and j are bonded
                    atom_i = mol.atoms[i-1]
                    is_bonded = False
                    for bond in atom_i.bonds:
                        other = bond.other_end(atom_i)
                        other_idx = mol.atoms.index(other) + 1
                        if other_idx == j:
                            is_bonded = True
                            break
                    row += "  X " if is_bonded else "  . "
            print(row)
        print("")

    def _count_atoms_near_cp(
        self,
        mol: Molecule,
        cp_coords: np.ndarray,
        max_ring_distance: float = 3.0,
        max_oxygen_distance: float = 3.8
    ) -> Tuple[int, int]:
        """
        Count carbon and oxygen atoms near a critical point.

        This method is used to validate ring critical points by checking
        if they have the expected number of nearby atoms.

        Args:
            mol: PLAMS Molecule object
            cp_coords: Coordinates of the critical point [x, y, z]
            max_ring_distance: Maximum distance for carbon atoms from CP (Å)
            max_oxygen_distance: Maximum distance for oxygen atoms from CP (Å)

        Returns:
            Tuple of (carbon_count, oxygen_count)
        """
        carbon_count = 0
        oxygen_count = 0

        for atom in mol.atoms:
            atom_coords = np.array([atom.x, atom.y, atom.z])
            distance = np.linalg.norm(atom_coords - cp_coords)

            if atom.symbol == 'C' and distance <= max_ring_distance:
                carbon_count += 1
            elif atom.symbol == 'O' and distance <= max_oxygen_distance:
                oxygen_count += 1

        return carbon_count, oxygen_count

    def _generate_threshold_combinations(
        self,
        base_C: float,
        base_O: float,
        increment: float,
        max_distance: int
    ) -> List[Tuple[int, float, float]]:
        """
        Generate threshold combinations by Manhattan distance.

        Explores the 2D space of (C_threshold, O_threshold) by increasing
        Manhattan distance from the base values. This allows independent
        adjustment of C and O thresholds.

        Args:
            base_C: Base carbon threshold (default: 3.0 Å)
            base_O: Base oxygen threshold (default: 3.8 Å)
            increment: Step size for adjustments (default: 0.1 Å)
            max_distance: Maximum Manhattan distance to explore

        Returns:
            List of (manhattan_distance, C_threshold, O_threshold) tuples
            ordered by increasing Manhattan distance
        """
        combinations = []

        # Explore by Manhattan distance layers
        for manhattan_dist in range(max_distance + 1):
            layer_combinations = []

            # For each Manhattan distance, generate all (delta_C, delta_O) pairs
            # where |delta_C| + |delta_O| = manhattan_dist
            for delta_C in range(-manhattan_dist, manhattan_dist + 1):
                remaining = manhattan_dist - abs(delta_C)

                # Two cases: +remaining and -remaining for delta_O
                for delta_O in ([remaining, -remaining] if remaining > 0 else [0]):
                    C_threshold = base_C + delta_C * increment
                    O_threshold = base_O + delta_O * increment

                    # Only keep positive thresholds
                    if C_threshold > 0 and O_threshold > 0:
                        layer_combinations.append((manhattan_dist, C_threshold, O_threshold))

            # Add layer combinations (avoid duplicates within same distance)
            seen = set()
            for combo in layer_combinations:
                key = (combo[1], combo[2])  # (C, O) pair
                if key not in seen:
                    seen.add(key)
                    combinations.append(combo)

        return combinations

    def find_ring_critical_point(
        self,
        ring_cps: List[Dict],
        ring_size: int = 6,
        mol: Optional[Molecule] = None,
        expected_oxygens: int = 2,
        max_ring_distance: float = 3.0,
        max_oxygen_distance: float = 3.8,
        distance_increment: float = 0.1,
        max_iterations: int = 10
    ) -> Dict:
        """
        Select the correct ring critical point among multiple candidates.

        **Strict validation strategy (when mol is provided):**
        1. Test ALL ring CPs to count nearby atoms (expected: 6C + 2O)
        2. If exactly 1 rCP matches → selection
        3. If 0 rCP match → increase distance thresholds and retry (up to max_iterations)
        4. If 2+ rCP match → raise ValueError (ambiguity, multiple rings match)

        **Adaptive distance thresholds:**
        - If no rCP matches initially, distance thresholds are incrementally increased
        - Increment: distance_increment Å per iteration (default: 0.1 Å)
        - Stops when: 1 rCP matches, 2+ rCPs match (ambiguity), or max_iterations reached

        **Legacy strategy (when mol is not provided, backward compatibility):**
        - Select the one with highest density

        Args:
            ring_cps: List of ring critical points from QTAIM
            ring_size: Number of expected carbon atoms (default: 6)
            mol: PLAMS Molecule object (optional, for geometric validation)
            expected_oxygens: Number of expected oxygen atoms (default: 2)
            max_ring_distance: Initial maximum distance CP-carbon (default: 3.0 Å)
            max_oxygen_distance: Initial maximum distance CP-oxygen (default: 3.8 Å)
            distance_increment: Distance increment per iteration (default: 0.1 Å)
            max_iterations: Maximum number of iterations for adaptive thresholds (default: 10)

        Returns:
            Selected ring CP dictionary

        Raises:
            ValueError: If no ring CPs found, max iterations reached, or ambiguity detected
        """
        if len(ring_cps) == 0:
            raise ValueError("No ring critical points found in RKF file")

        # Simple case: only one rCP detected
        if len(ring_cps) == 1:
            self.logger.info("Only one ring CP found → selected by default")
            return ring_cps[0]

        # Complex case: multiple rCPs detected
        self.logger.info(f"Found {len(ring_cps)} ring critical points")

        # STRICT GEOMETRIC VALIDATION WITH 2D THRESHOLD EXPLORATION (if mol provided)
        if mol is not None:
            # Generate all threshold combinations by Manhattan distance
            # max_iterations now represents max Manhattan distance (default: 10)
            max_manhattan_distance = max_iterations
            combinations = self._generate_threshold_combinations(
                max_ring_distance, max_oxygen_distance,
                distance_increment, max_manhattan_distance
            )

            self.logger.info(
                f"Testing thresholds by Manhattan distance (max_distance={max_manhattan_distance}, "
                f"{len(combinations)} combinations)..."
            )

            # Group combinations by Manhattan distance for logging
            current_distance = -1

            # Test each combination
            for manhattan_dist, C_threshold, O_threshold in combinations:
                # Log distance header when entering new distance layer
                if manhattan_dist != current_distance:
                    current_distance = manhattan_dist
                    self.logger.info(f"Distance {manhattan_dist}: Exploring C and O thresholds...")

                # Log current thresholds being tested
                self.logger.info(f"  Testing C≤{C_threshold:.1f}Å, O≤{O_threshold:.1f}Å")

                validated_cps = []

                # Test ALL rCPs with current thresholds
                for i, cp in enumerate(ring_cps, start=1):
                    cp_coords = np.array(cp['coords'])
                    density = cp.get('density', 0.0)

                    # Count atoms near this rCP
                    n_carbons, n_oxygens = self._count_atoms_near_cp(
                        mol, cp_coords, C_threshold, O_threshold
                    )

                    matches = (n_carbons == ring_size and n_oxygens == expected_oxygens)
                    marker = "✓ VALID" if matches else "✗ invalid"

                    self.logger.info(
                        f"    rCP {i} (index {cp['index']}): "
                        f"ρ={density:.6f}, {n_carbons}C + {n_oxygens}O → {marker}"
                    )

                    if matches:
                        validated_cps.append((i, cp))

                # Check validation result
                n_valid = len(validated_cps)

                if n_valid == 1:
                    # ✓ Success: exactly one rCP matches
                    idx, selected = validated_cps[0]
                    self.logger.info(
                        f"✓ Unique match found at Manhattan distance {manhattan_dist}: "
                        f"rCP {idx} (index {selected['index']}) → {ring_size}C + {expected_oxygens}O "
                        f"(thresholds: C≤{C_threshold:.1f}Å, O≤{O_threshold:.1f}Å)"
                    )
                    return selected

                elif n_valid >= 2:
                    # ✗ Ambiguity detected - stop immediately
                    indices = [idx for idx, _ in validated_cps]
                    self.logger.error(
                        f"✗ Ambiguity detected: {n_valid} ring CPs match {ring_size}C + {expected_oxygens}O "
                        f"at Manhattan distance {manhattan_dist} with thresholds C≤{C_threshold:.1f}Å, O≤{O_threshold:.1f}Å "
                        f"(rCP indices: {indices})"
                    )
                    raise ValueError(
                        f"Ambiguous ring topology: {n_valid} ring critical points match "
                        f"the expected {ring_size}C + {expected_oxygens}O pattern "
                        f"with distance thresholds C≤{C_threshold:.1f}Å, O≤{O_threshold:.1f}Å. "
                        f"Cannot determine which ring is the target. "
                        f"Matching rCP indices: {[cp['index'] for _, cp in validated_cps]}"
                    )

                # n_valid == 0: No match with this combination, continue to next

            # All combinations exhausted without finding a unique match
            self.logger.error(
                f"✗ No ring CP gives exactly {ring_size}C + {expected_oxygens}O "
                f"after exploring {len(combinations)} threshold combinations "
                f"up to Manhattan distance {max_manhattan_distance}"
            )
            raise ValueError(
                f"No ring critical point matches the expected topology "
                f"({ring_size}C + {expected_oxygens}O) after exploring {len(combinations)} combinations. "
                f"Explored C thresholds: {max_ring_distance - max_manhattan_distance * distance_increment:.1f} to "
                f"{max_ring_distance + max_manhattan_distance * distance_increment:.1f} Å, "
                f"O thresholds: {max_oxygen_distance - max_manhattan_distance * distance_increment:.1f} to "
                f"{max_oxygen_distance + max_manhattan_distance * distance_increment:.1f} Å. "
                f"Found {len(ring_cps)} rCP(s) but none have the correct atom count."
            )

        # FALLBACK: Density-based selection (backward compatibility, mol not provided)
        self.logger.warning("No molecule provided for validation, using density-based selection")

        if 'density' in ring_cps[0]:
            sorted_cps = sorted(ring_cps, key=lambda cp: cp['density'], reverse=True)

            for i, cp in enumerate(sorted_cps):
                marker = "✓ SELECTED" if i == 0 else ""
                coords = cp['coords']
                self.logger.info(
                    f"  rCP {i+1} (index {cp['index']}): ρ = {cp['density']:.6f}, "
                    f"coords = ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}) {marker}"
                )

            selected = sorted_cps[0]
            self.logger.info(f"Selected rCP with highest density: ρ = {selected['density']:.6f}")
            return selected

        # Extreme case: no density information available
        self.logger.warning("No density information available, using first rCP")
        return ring_cps[0]
    
    def find_ring_and_oxygen_atoms(
        self,
        mol: Molecule,
        cp_coords: np.ndarray,
        max_ring_distance: float = 3.0,
        max_oxygen_distance: float = 3.8
    ) -> Tuple[List[int], List[int]]:
        """
        Find the 6 carbon atoms of the ring and 2 closest oxygen atoms.

        Args:
            mol: PLAMS Molecule object
            cp_coords: Coordinates of the ring critical point [x, y, z]
            max_ring_distance: Maximum distance for ring atoms from CP (Å)
            max_oxygen_distance: Maximum distance for oxygen atoms from CP (Å)

        Returns:
            Tuple of (ring_atom_indices, oxygen_atom_indices)
        """
        self.logger.info(f"CP coordinates used for distance calculation: ({cp_coords[0]:.3f}, {cp_coords[1]:.3f}, {cp_coords[2]:.3f})")
        self.logger.info(f"Distance thresholds: Carbon ≤ {max_ring_distance:.1f} Å, Oxygen ≤ {max_oxygen_distance:.1f} Å")

        carbon_distances = []
        oxygen_distances = []

        for i, atom in enumerate(mol.atoms, start=1):  # PLAMS uses 1-based indexing
            atom_coords = np.array([atom.x, atom.y, atom.z])
            distance = np.linalg.norm(atom_coords - cp_coords)

            if atom.symbol == 'C' and distance <= max_ring_distance:
                carbon_distances.append((i, distance, atom_coords))
            elif atom.symbol == 'O' and distance <= max_oxygen_distance:
                oxygen_distances.append((i, distance, atom_coords))

        self.logger.info(f"Found {len(carbon_distances)} carbon(s) within {max_ring_distance:.1f} Å")
        if self.verbose:
            for idx, dist, coords in carbon_distances:
                self.logger.info(f"  C{idx}: distance = {dist:.3f} Å, coords = ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f})")

        # Sort by distance and take the 6 closest carbons
        carbon_distances.sort(key=lambda x: x[1])
        ring_atoms = [idx for idx, _, _ in carbon_distances[:6]]

        # Sort by distance and take the 2 closest oxygens
        oxygen_distances.sort(key=lambda x: x[1])
        oxygen_atoms = [idx for idx, _, _ in oxygen_distances[:2]]

        if len(ring_atoms) != 6:
            raise ValueError(f"Expected 6 ring carbons, found {len(ring_atoms)}")

        if len(oxygen_atoms) != 2:
            raise ValueError(f"Expected 2 oxygen atoms, found {len(oxygen_atoms)}")

        self.logger.info(f"Selected 6 closest carbons for ring: {ring_atoms}")
        self.logger.info(f"Selected 2 closest oxygens: {oxygen_atoms}")

        # Diagnostic: Print ring connectivity matrix (if verbose)
        if self.verbose:
            self._print_ring_connectivity(mol, ring_atoms)

        return ring_atoms, oxygen_atoms
    
    def get_bonded_atoms(
        self,
        mol: Molecule,
        atom_idx: int,
        allowed_atoms: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Get atoms bonded to a given atom.

        Args:
            mol: PLAMS Molecule object
            atom_idx: Atom index (1-based)
            allowed_atoms: Set of allowed atom indices (None = all)

        Returns:
            List of bonded atom indices
        """
        atom = mol.atoms[atom_idx - 1]  # Convert to 0-based for PLAMS
        bonded = []

        for bond in atom.bonds:
            other = bond.other_end(atom)
            other_idx = mol.atoms.index(other) + 1  # Convert to 1-based

            if allowed_atoms is None or other_idx in allowed_atoms:
                bonded.append(other_idx)

        return bonded
    
    def find_c1_carbon(
        self,
        mol: Molecule,
        ring_atoms: List[int]
    ) -> int:
        """
        Find C1: the carbon in the ring with 3 bonded carbons (branching point).
        
        Args:
            mol: PLAMS Molecule object
            ring_atoms: List of ring carbon atom indices
            
        Returns:
            Index of C1 carbon
        """
        ring_set = set(ring_atoms)

        for atom_idx in ring_atoms:
            bonded = self.get_bonded_atoms(mol, atom_idx)

            # Count carbons bonded to this atom
            carbon_bonds = [b for b in bonded if mol.atoms[b-1].symbol == 'C']

            # C1 should have 3 carbon bonds (2 in ring + 1 branch)
            if len(carbon_bonds) == 3:
                self.logger.info(f"Found C1 at index {atom_idx}")
                return atom_idx

        raise ValueError("Could not find C1 (branching carbon)")
    
    def find_c5_carbon(
        self,
        mol: Molecule,
        c1_idx: int,
        ring_atoms: List[int],
        oxygen_atoms: List[int]
    ) -> int:
        """
        Find C5: carbon at 2 bonds from C1 that is bonded to an oxygen.
        
        Args:
            mol: PLAMS Molecule object
            c1_idx: Index of C1
            ring_atoms: List of ring carbon atom indices
            oxygen_atoms: List of oxygen atom indices
            
        Returns:
            Index of C5 carbon
        """
        ring_set = set(ring_atoms)
        oxygen_set = set(oxygen_atoms)
        
        # Find carbons at distance 2 from C1 (in ring)
        c1_neighbors = self.get_bonded_atoms(mol, c1_idx, ring_set)
        
        candidates = []
        for neighbor in c1_neighbors:
            neighbor_neighbors = self.get_bonded_atoms(mol, neighbor, ring_set)
            for candidate in neighbor_neighbors:
                if candidate != c1_idx and candidate in ring_set:
                    # Check if this candidate is bonded to oxygen
                    candidate_bonds = self.get_bonded_atoms(mol, candidate)
                    if any(b in oxygen_set for b in candidate_bonds):
                        candidates.append(candidate)
        
        if len(candidates) == 0:
            raise ValueError("Could not find C5 (carbon at distance 2 from C1 bonded to O)")
        
        if len(candidates) > 1:
            self.logger.warning(f"Multiple C5 candidates found: {candidates}, using first")
        
        c5_idx = candidates[0]
        self.logger.info(f"Found C5 at index {c5_idx}")
        return c5_idx
    
    def trace_ring_path(
        self,
        mol: Molecule,
        c1_idx: int,
        c5_idx: int,
        ring_atoms: List[int]
    ) -> List[int]:
        """
        Trace the ring path: C5 -> C4 -> C3 -> C2 -> C1 -> C6 -> C5.
        
        Args:
            mol: PLAMS Molecule object
            c1_idx: Index of C1
            c5_idx: Index of C5
            ring_atoms: List of ring carbon atom indices
            
        Returns:
            Ordered list of ring atom indices [C1, C2, C3, C4, C5, C6]
        """
        ring_set = set(ring_atoms)
        
        # Start from C5 and trace back to C1
        path_to_c1 = [c5_idx]
        current = c5_idx
        previous = None
        
        while current != c1_idx:
            neighbors = self.get_bonded_atoms(mol, current, ring_set)
            # Find next neighbor (not previous)
            next_atoms = [n for n in neighbors if n != previous and n in ring_set]
            
            if len(next_atoms) == 0:
                raise ValueError(f"Ring path broken at atom {current}")
            
            # Choose the path towards C1
            found = False
            for next_atom in next_atoms:
                if next_atom == c1_idx:
                    path_to_c1.append(next_atom)
                    found = True
                    break
            
            if not found:
                # Continue with first available neighbor
                previous = current
                current = next_atoms[0]
                path_to_c1.append(current)
            else:
                break
        
        # Now trace the OTHER side of the ring: C5 -> C6 -> ... -> C1
        # C6 is the neighbor of C5 that is NOT in path_to_c1
        c5_neighbors = self.get_bonded_atoms(mol, c5_idx, ring_set)
        c6_candidates = [n for n in c5_neighbors if n not in path_to_c1]

        if len(c6_candidates) != 1:
            raise ValueError(f"Expected 1 C6 candidate (other neighbor of C5), found {len(c6_candidates)}")

        c6_idx = c6_candidates[0]

        # Now trace from C6 back to C1 to get the other side of the ring
        path_from_c6_to_c1 = [c6_idx]
        current = c6_idx
        previous = c5_idx  # We came from C5

        while current != c1_idx:
            neighbors = self.get_bonded_atoms(mol, current, ring_set)
            # Find next neighbor (not previous, not in path_to_c1 except C1)
            next_atoms = [n for n in neighbors if n != previous and (n == c1_idx or n not in path_to_c1)]

            if len(next_atoms) == 0:
                raise ValueError(f"Ring path broken at atom {current} while tracing C6 to C1")

            # Choose the path towards C1
            if c1_idx in next_atoms:
                path_from_c6_to_c1.append(c1_idx)
                break
            else:
                previous = current
                current = next_atoms[0]
                path_from_c6_to_c1.append(current)

        # Verify ring closure
        if path_from_c6_to_c1[-1] != c1_idx:
            raise ValueError("Ring path does not close properly (C6 side doesn't reach C1)")

        # Build final ordered path: [C1, C2, C3, ..., C5, C6]
        # path_to_c1 is [C5, ..., C1], reverse to get [C1, ..., C5]
        # path_from_c6_to_c1 is [C6, ..., C1], take all except final C1
        ordered_path = list(reversed(path_to_c1))
        ordered_path.extend(path_from_c6_to_c1[:-1])
        
        self.logger.info(f"Ring path traced: {ordered_path}")
        
        return ordered_path[:6]  # Return [C1, C2, C3, C4, C5, C6]
    
    def assign_oxygen_indices(
        self,
        mol: Molecule,
        ring_path: List[int],
        oxygen_atoms: List[int]
    ) -> Tuple[int, int]:
        """
        Assign O7 and O8 by finding which ring carbons are bonded to oxygen.

        Args:
            mol: PLAMS Molecule object
            ring_path: Ordered list of ring carbon indices [C1, C2, C3, C4, C5, C6]
            oxygen_atoms: List of oxygen atom indices

        Returns:
            Tuple of (O7_index, O8_index) where O7 is the first oxygen found
            and O8 is the second oxygen found (in ring path order)
        """
        oxygen_set = set(oxygen_atoms)

        # Find which ring carbons are bonded to oxygen
        carbons_with_oxygen = []
        for carbon_idx in ring_path:
            neighbors = self.get_bonded_atoms(mol, carbon_idx)
            oxygen_neighbors = [n for n in neighbors if n in oxygen_set]
            if oxygen_neighbors:
                for oxy_idx in oxygen_neighbors:
                    carbons_with_oxygen.append((carbon_idx, oxy_idx))

        if len(carbons_with_oxygen) != 2:
            raise ValueError(f"Expected 2 carbons bonded to oxygen, found {len(carbons_with_oxygen)}")

        # Assign O7 to first oxygen found (in ring path order), O8 to second
        o7_idx = carbons_with_oxygen[0][1]
        o8_idx = carbons_with_oxygen[1][1]

        self.logger.info(f"Assigned O7={o7_idx}, O8={o8_idx}")

        return o7_idx, o8_idx
    
    def reindex_atoms(
        self,
        mol: Molecule,
        cp_coords: np.ndarray
    ) -> Dict[int, int]:
        """
        Perform complete atom reindexing based on molecular topology.

        Args:
            mol: PLAMS Molecule object
            cp_coords: Coordinates of ring critical point

        Returns:
            Dictionary mapping canonical_index -> original_index
        """
        # Diagnostic: Print complete molecular structure and connectivity (if verbose)
        if self.verbose:
            self._print_molecular_structure(mol)
            self._print_connectivity_graph(mol)

        # Step 1: Find ring and oxygen atoms
        ring_atoms, oxygen_atoms = self.find_ring_and_oxygen_atoms(mol, cp_coords)
        
        # Step 2: Find C1 (branching carbon)
        c1_idx = self.find_c1_carbon(mol, ring_atoms)
        
        # Step 3: Find C5 (carbon at distance 2 from C1 bonded to O)
        c5_idx = self.find_c5_carbon(mol, c1_idx, ring_atoms, oxygen_atoms)
        
        # Step 4: Trace ring path to get ordered carbons
        ring_path = self.trace_ring_path(mol, c1_idx, c5_idx, ring_atoms)

        # ring_path should be [C1, C2, C3, C4, C5, C6]
        c1, c2, c3, c4, c5, c6 = ring_path

        # Step 5: Assign oxygen indices (automatically find which carbons are bonded to O)
        o7_idx, o8_idx = self.assign_oxygen_indices(mol, ring_path, oxygen_atoms)
        
        # Step 6: Create reindex maps
        # Canonical: C1=1, C2=2, C3=3, C4=4, C5=5, C6=6, O7=7, O8=8
        self.reverse_map = {
            1: c1, 2: c2, 3: c3, 4: c4,
            5: c5, 6: c6, 7: o7_idx, 8: o8_idx
        }
        
        self.reindex_map = {v: k for k, v in self.reverse_map.items()}
        
        self.logger.info(f"Reindexing complete: {self.reindex_map}")
        
        return self.reverse_map
    
    def get_canonical_index(self, original_idx: int) -> Optional[int]:
        """Get canonical index from original index."""
        return self.reindex_map.get(original_idx)
    
    def get_original_index(self, canonical_idx: int) -> Optional[int]:
        """Get original index from canonical index."""
        return self.reverse_map.get(canonical_idx)

