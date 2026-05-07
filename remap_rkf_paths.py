#!/usr/bin/env python3
"""
Script temporaire pour remapper les chemins CRIANN vers les chemins locaux.

Transforme :
  /dlocal/run/<JOB_ID>/<mol>_workdir/...
en :
  <DATA_DIR>/<Solvent>/<mol>_workdir/...

Cibles :
  1. neutral_rkf_paths.json (conformers)
  2. Symlinks dans redox/results/rkf/
"""

import json
import os
import re
import sys
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Pattern pour extraire la partie relative depuis un chemin CRIANN
CRIANN_PATTERN = re.compile(r'/dlocal/run/\d+/(\S+_workdir/.*)')


def remap_path(criann_path, solvent_dir):
    """Remappe un chemin CRIANN vers le chemin local."""
    m = CRIANN_PATTERN.match(criann_path)
    if not m:
        return None
    relative = m.group(1)  # e.g. F-HMC_workdir/conformers/calculations/...
    return str(solvent_dir / relative)


def remap_json_files(data_dir):
    """Remappe tous les neutral_rkf_paths.json."""
    json_ok = 0
    json_err = 0
    json_skip = 0
    missing_rkf = []

    for json_path in sorted(data_dir.rglob("neutral_rkf_paths.json")):
        # Deduire le dossier solvant (data/<Solvent>/<mol>_workdir/conformers/results/)
        # solvent_dir = data/<Solvent>/
        parts = json_path.relative_to(data_dir).parts
        solvent = parts[0]
        solvent_dir = data_dir / solvent

        with open(json_path) as f:
            paths = json.load(f)

        modified = False
        new_paths = {}
        for key, old_path in paths.items():
            if '/dlocal/run/' not in old_path:
                new_paths[key] = old_path
                continue

            new_path = remap_path(old_path, solvent_dir)
            if new_path is None:
                print(f"  WARN: pattern non reconnu: {old_path}")
                new_paths[key] = old_path
                json_err += 1
                continue

            if not os.path.exists(new_path):
                missing_rkf.append(new_path)

            new_paths[key] = new_path
            modified = True

        if modified:
            with open(json_path, 'w') as f:
                json.dump(new_paths, f, indent=2)
            json_ok += 1
        else:
            json_skip += 1

    return json_ok, json_skip, json_err, missing_rkf


def remap_symlinks(data_dir):
    """Remappe tous les symlinks dans redox/results/rkf/."""
    sym_ok = 0
    sym_err = 0
    sym_skip = 0
    missing_targets = []

    for rkf_dir in sorted(data_dir.rglob("redox/results/rkf")):
        if not rkf_dir.is_dir():
            continue

        # solvent_dir
        parts = rkf_dir.relative_to(data_dir).parts
        solvent = parts[0]
        solvent_dir = data_dir / solvent

        for symlink in sorted(rkf_dir.iterdir()):
            if not symlink.is_symlink():
                continue

            target = os.readlink(symlink)
            if '/dlocal/run/' not in target:
                sym_skip += 1
                continue

            new_target = remap_path(target, solvent_dir)
            if new_target is None:
                print(f"  WARN: pattern non reconnu: {target}")
                sym_err += 1
                continue

            if not os.path.exists(new_target):
                missing_targets.append(new_target)

            # Recreer le symlink
            symlink.unlink()
            symlink.symlink_to(new_target)
            sym_ok += 1

    return sym_ok, sym_skip, sym_err, missing_targets


def main():
    if not DATA_DIR.exists():
        print(f"Erreur: {DATA_DIR} n'existe pas")
        sys.exit(1)

    print("=" * 70)
    print("Remapping des chemins CRIANN -> locaux")
    print("=" * 70)
    print(f"Data dir: {DATA_DIR}\n")

    # 1. JSON files
    print("--- neutral_rkf_paths.json ---")
    json_ok, json_skip, json_err, json_missing = remap_json_files(DATA_DIR)
    print(f"  Remappes: {json_ok}")
    print(f"  Ignores (deja locaux): {json_skip}")
    print(f"  Erreurs: {json_err}")
    if json_missing:
        print(f"  ATTENTION: {len(json_missing)} fichiers RKF manquants localement")
        for p in json_missing[:5]:
            print(f"    - {p}")
        if len(json_missing) > 5:
            print(f"    ... et {len(json_missing) - 5} de plus")

    # 2. Symlinks
    print("\n--- Symlinks redox/results/rkf/ ---")
    sym_ok, sym_skip, sym_err, sym_missing = remap_symlinks(DATA_DIR)
    print(f"  Remappes: {sym_ok}")
    print(f"  Ignores (deja locaux): {sym_skip}")
    print(f"  Erreurs: {sym_err}")
    if sym_missing:
        print(f"  ATTENTION: {len(sym_missing)} cibles manquantes localement")
        for p in sym_missing[:5]:
            print(f"    - {p}")
        if len(sym_missing) > 5:
            print(f"    ... et {len(sym_missing) - 5} de plus")

    print(f"\n{'=' * 70}")
    total = json_ok + sym_ok
    total_err = json_err + sym_err + len(json_missing) + len(sym_missing)
    print(f"Total remappes: {total} ({json_ok} JSON + {sym_ok} symlinks)")
    if total_err:
        print(f"Total problemes: {total_err}")
    else:
        print("Aucun probleme detecte !")


if __name__ == "__main__":
    main()
