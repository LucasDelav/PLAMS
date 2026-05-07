#!/usr/bin/env amspython
"""Generate 2D molecular structure PNGs from SMILES in input.txt.

Each molecule is rendered to fill its own frame (per-molecule bounds).
Output: SVG converted to PNG at default resolution.
"""

import os
import subprocess
import tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Geometry import rdGeometry

# ── Templates for alignment ──────────────────────────────────────────────────
CATECHOL_SMILES = "c1(c(cccc1)O)O"
QUINONE_SMILES = "C1(=O)C(=O)C=CC=C1"
INDOLE_CATECHOL_SMILES = "c1(c(cc2c(c1)cc[nH]2)O)O"
INDOLE_QUINONE_SMILES = "C1(=O)C(=O)C=c2c(=C1)cc[nH]2"


def _make_template(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return mol


TEMPLATE_CATECHOL = _make_template(CATECHOL_SMILES)
TEMPLATE_QUINONE = _make_template(QUINONE_SMILES)
TEMPLATE_INDOLE_CATECHOL = _make_template(INDOLE_CATECHOL_SMILES)
TEMPLATE_INDOLE_QUINONE = _make_template(INDOLE_QUINONE_SMILES)


# ── Input parsing ────────────────────────────────────────────────────────────
def parse_input(path):
    """Parse input.txt → list of (smiles, name)."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries


# ── Template selection & 2D coordinate generation ────────────────────────────
def select_template(mol):
    """Return (template, flip) for *mol*."""
    pat_indole = Chem.MolFromSmarts("[#6]~1~[#6]~[#6]~[#6]~2~[#6](~[#6]~1)~[#6]~[#6]~[*]~2")
    pat_quinone = Chem.MolFromSmarts("[#6](=O)[#6](=O)")
    pat_catechol = Chem.MolFromSmarts("c(cO)O")
    has_indole = mol.HasSubstructMatch(pat_indole)
    has_quinone = mol.HasSubstructMatch(pat_quinone)
    has_catechol = mol.HasSubstructMatch(pat_catechol)
    if has_indole and (has_quinone or has_catechol):
        return None, True
    if has_quinone:
        return TEMPLATE_QUINONE, False
    if has_catechol:
        return TEMPLATE_CATECHOL, False
    return None, False


def prepare_mol(mol):
    """Compute 2D coords for *mol* with template alignment."""
    template, flip_coords = select_template(mol)
    if flip_coords:
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
        cx = (max(xs) + min(xs)) / 2.0
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (2 * cx - p.x, p.y, 0.0))
    elif template is not None:
        try:
            AllChem.GenerateDepictionMatching2DStructure(mol, template)
        except Exception:
            AllChem.Compute2DCoords(mol)
    else:
        AllChem.Compute2DCoords(mol)


# ── Rendering ────────────────────────────────────────────────────────────────
PNG_WIDTH = 400
PNG_HEIGHT = 300
BOND_LINE_WIDTH = 2.5
COORD_PAD = 2.0  # padding in coordinate space around atoms (for labels)


def _normalize_bond_length(mol, target=1.5):
    """Scale 2D coordinates so the average bond length equals *target*."""
    conf = mol.GetConformer()
    lengths = []
    for bond in mol.GetBonds():
        p1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        p2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
        lengths.append(np.hypot(p2.x - p1.x, p2.y - p1.y))
    if not lengths:
        return
    avg = np.mean(lengths)
    if avg < 1e-6:
        return
    ratio = target / avg
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x * ratio, p.y * ratio, 0.0))


def _mol_bbox(mol):
    """Return (width, height) in coordinate space (with padding)."""
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    return (max(xs) - min(xs) + 2 * COORD_PAD,
            max(ys) - min(ys) + 2 * COORD_PAD)


def compute_uniform_scale(mols):
    """Find the largest scale (px/coord-unit) that fits every molecule in
    the PNG canvas.  All molecules must already be normalised."""
    max_w, max_h = 0.0, 0.0
    for mol in mols:
        if mol is None:
            continue
        w, h = _mol_bbox(mol)
        max_w = max(max_w, w)
        max_h = max(max_h, h)
    # scale so that the biggest molecule just fits
    return min(PNG_WIDTH / max_w, PNG_HEIGHT / max_h)


def render_molecule(mol, name, output_dir, scale):
    """Render *mol* at the given fixed *scale* (px per coord unit).

    Every molecule uses exactly the same scale, so bond lengths, line
    widths, font sizes and dash patterns are pixel-identical.  The
    molecule SVG is rendered at its natural pixel size, then composited
    centered onto the final 400×300 canvas.
    """
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    x_min, x_max = min(xs) - COORD_PAD, max(xs) + COORD_PAD
    y_min, y_max = min(ys) - COORD_PAD, max(ys) + COORD_PAD

    svg_w = max(int((x_max - x_min) * scale), 1)
    svg_h = max(int((y_max - y_min) * scale), 1)

    min_pt = rdGeometry.Point2D(x_min, y_min)
    max_pt = rdGeometry.Point2D(x_max, y_max)

    drawer = Draw.MolDraw2DSVG(svg_w, svg_h)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.useBWAtomPalette()
    opts.bondLineWidth = BOND_LINE_WIDTH
    opts.multipleBondOffset = 0.15
    opts.padding = 0.0

    drawer.SetScale(svg_w, svg_h, min_pt, max_pt)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Save SVG for debugging
    svg_path = os.path.join(output_dir, f"{name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg)

    png_path = os.path.join(output_dir, f"{name}.png")
    with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as tmp_f:
        tmp_f.write(svg)
        tmp_svg = tmp_f.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png_f:
        tmp_png = tmp_png_f.name

    try:
        # 1) SVG → PNG at native size (no rescaling)
        subprocess.run(
            ["rsvg-convert", "-f", "png", "-o", tmp_png, tmp_svg],
            check=True, capture_output=True,
        )
        # 2) Composite centered on a 400×300 transparent canvas
        subprocess.run(
            ["magick", tmp_png,
             "-gravity", "center",
             "-background", "none",
             "-extent", f"{PNG_WIDTH}x{PNG_HEIGHT}",
             png_path],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(tmp_svg)
        if os.path.exists(tmp_png):
            os.unlink(tmp_png)
    return png_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input.txt")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    entries = parse_input(input_path)
    print(f"Found {len(entries)} molecules in input.txt")

    # Pass 1 — parse SMILES, compute 2D coords, normalise bond lengths
    mols = []
    for smiles, name in entries:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [x] {name}: invalid SMILES '{smiles}'")
            mols.append(None)
            continue
        prepare_mol(mol)
        _normalize_bond_length(mol)
        mols.append(mol)

    # Pass 2 — determine the single scale that fits every molecule
    scale = compute_uniform_scale(mols)
    print(f"Uniform scale: {scale:.1f} px/coord-unit "
          f"(bond ≈ {1.5 * scale:.0f} px)")

    # Pass 3 — render all molecules with that scale
    ok, fail = 0, 0
    for (smiles, name), mol in zip(entries, mols):
        if mol is None:
            fail += 1
            continue
        try:
            png = render_molecule(mol, name, output_dir, scale)
            print(f"  [+] {name} -> {os.path.basename(png)}")
            ok += 1
        except Exception as e:
            print(f"  [x] {name}: {e}")
            fail += 1

    print(f"\nDone: {ok} generated, {fail} failed — output in {output_dir}/")


if __name__ == "__main__":
    main()
