"""
Pipeline orchestration for QTAIM/IQA calculations.

Ce module implémente le pipeline QTAIM/IQA pour l'analyse topologique
et la décomposition énergétique des conformères.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

from scm.plams import Molecule, AMSJob

from common.workspace import setup_workspace as common_setup_workspace, get_pipeline_dirs
from common.utils import print_success, print_error, print_section

from .common import (
    get_conformer_files,
    create_adf_settings,
    CalculationError
)
from .analysis import QTAIMAnalyzer
from .topology import TopologyAnalyzer


class QTAIMPipeline:
    """
    Pipeline for QTAIM/IQA calculations on conformers.
    """

    def __init__(
        self,
        workdir,
        functional="BLYP",
        basis="TZP",
        quality="Good",
        integration_accuracy=5.0,
        run_iqa=True,
        nproc=1,
        restart=False,
        molecule_name=None
    ):
        """
        Initialize QTAIM pipeline.

        Args:
            workdir (str or Path): Working directory
            functional (str): DFT functional
            basis (str): Basis set
            quality (str): Numerical quality
            integration_accuracy (float): QTAIM integration accuracy
            run_iqa (bool): Enable IQA calculations
            nproc (int): Number of parallel processes
            restart (bool): Restart from existing calculations
            molecule_name (str, optional): Molecule name (for logging context)
        """
        self.workdir = Path(workdir)
        self.functional = functional
        self.basis = basis
        self.quality = quality
        self.integration_accuracy = integration_accuracy
        self.run_iqa = run_iqa
        self.nproc = nproc
        self.restart = restart
        self.molecule_name = molecule_name

        # Setup directories using common function
        self.calc_dir, self.results_dir = get_pipeline_dirs(self.workdir, "qtaim")

        # Initialize analyzers
        self.analyzer = QTAIMAnalyzer()
        self.topology_analyzer = TopologyAnalyzer()

        print_success(f"QTAIM pipeline initialized in {self.workdir}")

    def get_conformers(self, conformer_ids=None):
        """
        Get conformer files to process.

        Args:
            conformer_ids (list, optional): Specific conformer IDs to process (None = all)

        Returns:
            dict: Dictionary mapping conformer_id -> xyz_file
        """
        conformers = get_conformer_files(self.workdir, state="neutral")

        if conformer_ids is not None:
            conformers = {
                cid: path for cid, path in conformers.items()
                if cid in conformer_ids
            }

        print_success(f"Found {len(conformers)} conformer(s) to process")
        return conformers

    def load_existing_calculations(self, conformer_ids=None, charges=[0, -1, +1]):
        """
        Load existing RKF files from previous calculations instead of re-running.

        This method allows re-analysis of QTAIM/IQA data without redoing expensive
        DFT calculations. Useful for changing analysis parameters or fixing bugs.

        Args:
            conformer_ids (list, optional): Specific conformer IDs to process (None = all)
            charges (list): List of charge states to load

        Returns:
            dict: {conformer_id: {charge: rkf_path}}

        Example:
            >>> pipeline = QTAIMPipeline(workdir="molecule_workdir")
            >>> rkf_files = pipeline.load_existing_calculations(charges=[0, -1])
            >>> # rkf_files = {'conf_1': {0: Path(...), -1: Path(...)}, ...}
        """
        # Get conformers to process
        conformers = self.get_conformers(conformer_ids)

        if not conformers:
            print_error("No conformers found!")
            return {}

        # Look for RKF files in plams_workdir
        calc_dir = self.calc_dir / "plams_workdir"

        if not calc_dir.exists():
            print_error(f"Calculations directory not found: {calc_dir}")
            return {}

        results = {}

        # Charge labels mapping
        charge_labels = {0: "neutral", -1: "reduced", +1: "oxidized"}

        for conf_id in conformers.keys():
            results[conf_id] = {}

            for charge in charges:
                charge_label = charge_labels.get(charge, f"charge{charge:+d}")

                # Pattern for job directories: {molecule_name}_{conf_id}_{charge_label}
                if self.molecule_name:
                    pattern = f"{self.molecule_name}_{conf_id}_{charge_label}*"
                else:
                    pattern = f"*{conf_id}_{charge_label}*"

                # Find matching job directories
                job_dirs = list(calc_dir.glob(pattern))

                if job_dirs:
                    # Take the first (most recent if multiple)
                    rkf_path = job_dirs[0] / "adf.rkf"

                    if rkf_path.exists():
                        results[conf_id][charge] = rkf_path
                        print(f"  Found {conf_id} {charge_label}: {rkf_path.name}")
                    else:
                        print_error(f"  Missing RKF for {conf_id} {charge_label}")
                else:
                    print(f"  ⚠ No calculation found for {conf_id} {charge_label}")

        # Summary
        total_found = sum(len(charges_dict) for charges_dict in results.values())
        expected = len(conformers) * len(charges)
        print_success(f"Loaded {total_found}/{expected} existing calculations")

        return results

    def run_calculation(self, conformer_id, xyz_file, charge=0):
        """
        Run QTAIM/IQA calculation for a conformer with specific charge.

        Args:
            conformer_id (str): Conformer identifier
            xyz_file (Path): Path to XYZ geometry file
            charge (int): Molecular charge

        Returns:
            Optional[Path]: Path to RKF file or None if calculation failed
        """
        # Determine spin polarization (simple heuristic)
        spin_polarization = abs(charge) if charge != 0 else 0

        # Job name
        if charge == +1:
            charge_label = "oxidized"
        elif charge == -1:
            charge_label = "reduced"
        elif charge == 0:
            charge_label = "neutral"
        else:
            charge_label = "unknown"

        # Extract molecule name without _workdir suffix
        molecule_name = self.workdir.name.replace('_workdir', '')
        job_name = f"{molecule_name}_{conformer_id}_{charge_label}"

        print(f"  → Running {job_name}...")

        try:
            # Load molecule
            mol = Molecule(str(xyz_file))

            # Create settings
            settings = create_adf_settings(
                functional=self.functional,
                basis=self.basis,
                quality=self.quality,
                integration_accuracy=self.integration_accuracy,
                run_iqa=self.run_iqa,
                charge=charge,
                spin_polarization=spin_polarization
            )

            # Create and run job
            job = AMSJob(
                molecule=mol,
                settings=settings,
                name=job_name
            )

            result = job.run()

            if not result.ok():
                print_error(f"Job failed for {job_name}")
                return None

            # Get RKF file path
            rkf_path = Path(result.job.path) / "adf.rkf"

            if not rkf_path.exists():
                print_error(f"RKF file not found for {job_name}")
                return None

            print_success(f"Calculation completed for {job_name}")
            return rkf_path

        except Exception as e:
            print_error(f"Error in calculation for {job_name}: {str(e)}")
            return None

    def run_pipeline(self, conformer_ids=None, charges=[0, -1, +1], analysis_only=False):
        """
        Run complete QTAIM/IQA pipeline on conformers.

        Args:
            conformer_ids (list, optional): Specific conformer IDs to process (None = all)
            charges (list): List of charge states to calculate
            analysis_only (bool): If True, load existing RKF files instead of running calculations

        Returns:
            dict: Dictionary with results: {conformer_id: {charge: results}}
        """
        print_section("QTAIM/IQA PIPELINE")

        # ANALYSIS-ONLY MODE: Load existing calculations
        if analysis_only:
            print("Mode: ANALYSIS-ONLY (loading existing RKF files)")
            print("-" * 80)

            rkf_files = self.load_existing_calculations(conformer_ids, charges)

            if not rkf_files:
                print_error("No existing calculations found!")
                return {}

            # Get conformer files for molecular structure info
            conformers = self.get_conformers(conformer_ids)

            all_results = {}

            for conf_id, charge_rkfs in rkf_files.items():
                print(f"\nAnalyzing conformer: {conf_id}")

                if conf_id not in conformers:
                    print_error(f"  XYZ file not found for {conf_id}")
                    continue

                xyz_file = conformers[conf_id]
                mol = Molecule(str(xyz_file))
                mol.guess_bonds()  # Required for topology analysis

                conf_results = {}

                for charge, rkf_path in charge_rkfs.items():
                    print(f"  Processing charge {charge:+d}...")

                    results = self.analyzer.analyze_rkf(
                        rkf_path,
                        mol=mol,
                        topology_analyzer=self.topology_analyzer,
                        conformer_id=conf_id,
                        charge=charge,
                        molecule_name=self.molecule_name
                    )
                    conf_results[charge] = results

                all_results[conf_id] = conf_results

            print("\n" + "=" * 80)
            print_success(f"Analysis completed for {len(all_results)} conformer(s)")
            print("=" * 80)

            return all_results

        # CALCULATION MODE: Run full pipeline
        # Get conformers to process
        conformers = self.get_conformers(conformer_ids)

        if not conformers:
            print_error("No conformers found!")
            return {}

        print(f"\nProcessing {len(conformers)} conformer(s) with charges: {charges}")
        print("-" * 80)

        # Store all results
        all_results = {}

        # Process each conformer
        for conf_id, xyz_file in conformers.items():
            print(f"\nConformer: {conf_id}")

            conf_results = {}

            # Run calculations for each charge state
            for charge in charges:
                rkf_path = self.run_calculation(conf_id, xyz_file, charge)

                if rkf_path is not None:
                    # Analyze results with reindexing
                    mol = Molecule(str(xyz_file))
                    # Force bond detection for topology analysis
                    mol.guess_bonds()
                    results = self.analyzer.analyze_rkf(
                        rkf_path,
                        mol=mol,
                        topology_analyzer=self.topology_analyzer,
                        conformer_id=conf_id,
                        charge=charge,
                        molecule_name=self.molecule_name
                    )
                    conf_results[charge] = results

            all_results[conf_id] = conf_results

        print("\n" + "=" * 80)
        print_success(f"Pipeline completed for {len(all_results)} conformer(s)")
        print("=" * 80)

        return all_results
