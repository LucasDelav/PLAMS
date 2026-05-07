#!/usr/bin/env python3
"""
RKF Viewer GUI - Navigation sequentielle des fichiers RKF redox avec amsview.

Panneau de controle flottant (always-on-top) qui pilote amsview.

Usage:
    ./rkf_viewer.py data/                                    # Tout scanner
    ./rkf_viewer.py data/Acetonitrile/                       # Un solvant
    ./rkf_viewer.py data/Acetonitrile/Al-CA_workdir/         # Une molecule
    ./rkf_viewer.py data/Hexane/Al-CA_workdir/redox/results/rkf/  # Dossier RKF direct
"""

import argparse
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path


# Chemin vers amsbashrc.sh
_AMS_BASHRC = Path.home() / "ams2025.102" / "amsbashrc.sh"

# Calculs avec geometrie optimisee uniquement (ordre pipeline)
CALC_ORDER = {
    "neutral_DZP": 0,       # Geom opt neutre (depuis conformeres)
    "reduced_opt_DZP": 1,   # Geom opt reduit
    "oxidized_opt_DZP": 2,  # Geom opt oxyde
}


def get_ams_env():
    """Recupere l'environnement AMS en sourcant amsbashrc.sh si necessaire."""
    if os.environ.get("AMSBIN"):
        return None  # Environnement deja configure, utiliser l'actuel

    if not _AMS_BASHRC.exists():
        print(f"Erreur: {_AMS_BASHRC} introuvable et $AMSBIN non defini.")
        sys.exit(1)

    result = subprocess.run(
        ["bash", "-c", f"source {_AMS_BASHRC} && env"],
        capture_output=True, text=True
    )
    env = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key and _ == "=":
            env[key] = value
    return env


def parse_rkf_info(filepath):
    """Extrait solvant, molecule, conformere et type de calcul depuis le chemin du RKF."""
    path = Path(filepath)
    parts = path.parts

    workdir_idx = None
    for i, part in enumerate(parts):
        if part.endswith("_workdir"):
            workdir_idx = i
            break

    if workdir_idx is None:
        return None

    molecule = parts[workdir_idx].replace("_workdir", "")
    solvent = parts[workdir_idx - 1] if workdir_idx > 0 else "Unknown"

    filename = path.stem
    match = re.match(r"conf_(\d+)_(.+)", filename)
    if not match:
        return None

    return {
        "path": str(filepath),
        "solvent": solvent,
        "molecule": molecule,
        "conf_num": int(match.group(1)),
        "calc_type": match.group(2),
        "filename": path.name,
    }


def scan_rkf_files(base_path):
    """Scanne recursivement pour trouver les fichiers RKF dans redox/results/rkf/."""
    base = Path(base_path).resolve()
    rkf_files = []

    for rkf in base.rglob("*.rkf"):
        rkf_str = str(rkf)
        if "/redox/results/rkf/" in rkf_str or rkf_str.count("/rkf") > 0:
            info = parse_rkf_info(rkf)
            if info and info["calc_type"] in CALC_ORDER:
                rkf_files.append(info)

    return rkf_files


def sort_rkf_files(files):
    """Trie par solvant > molecule > conformere > ordre pipeline redox."""
    def sort_key(f):
        calc_order = CALC_ORDER.get(f["calc_type"], 99)
        return (f["solvent"], f["molecule"], f["conf_num"], calc_order)

    return sorted(files, key=sort_key)


class RKFViewerApp(tk.Tk):
    def __init__(self, rkf_files, ams_env):
        super().__init__()
        self.rkf_files = rkf_files
        self.ams_env = ams_env
        self.current_idx = 0
        self.viewer_process = None

        self.title("RKF Viewer")
        self.configure(bg="#1e1e1e")
        self.attributes("-topmost", True)
        self.resizable(True, False)

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Positionner en haut centre de l'ecran
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        win_w = 700
        x = (screen_w - win_w) // 2
        self.geometry(f"{win_w}x70+{x}+0")

        self._update_display()
        self._launch_viewer()

        # Forcer le focus clavier au demarrage (necessaire sous Hyprland/Wayland)
        self.after(200, self.focus_force)

    def _build_ui(self):
        # Ligne 1 : Molecule | Solvant | Conformere | Calcul
        row1 = tk.Frame(self, bg="#1e1e1e", padx=8, pady=3)
        row1.pack(fill=tk.X)

        self.lbl_molecule = tk.Label(
            row1, text="", font=("monospace", 13, "bold"),
            fg="#e0e0e0", bg="#1e1e1e"
        )
        self.lbl_molecule.pack(side=tk.LEFT)

        self.lbl_sep1 = tk.Label(
            row1, text="  |  ", font=("monospace", 13),
            fg="#555555", bg="#1e1e1e"
        )
        self.lbl_sep1.pack(side=tk.LEFT)

        self.lbl_solvent = tk.Label(
            row1, text="", font=("monospace", 13),
            fg="#8cb4ff", bg="#1e1e1e"
        )
        self.lbl_solvent.pack(side=tk.LEFT)

        self.lbl_sep2 = tk.Label(
            row1, text="  |  ", font=("monospace", 13),
            fg="#555555", bg="#1e1e1e"
        )
        self.lbl_sep2.pack(side=tk.LEFT)

        self.lbl_conf = tk.Label(
            row1, text="", font=("monospace", 13),
            fg="#a0a0a0", bg="#1e1e1e"
        )
        self.lbl_conf.pack(side=tk.LEFT)

        self.lbl_sep3 = tk.Label(
            row1, text="  |  ", font=("monospace", 13),
            fg="#555555", bg="#1e1e1e"
        )
        self.lbl_sep3.pack(side=tk.LEFT)

        self.lbl_calc = tk.Label(
            row1, text="", font=("monospace", 13, "bold"),
            fg="#ffcc66", bg="#1e1e1e"
        )
        self.lbl_calc.pack(side=tk.LEFT)

        # Ligne 2 : Navigation
        row2 = tk.Frame(self, bg="#1e1e1e", padx=8, pady=2)
        row2.pack(fill=tk.X)

        self.btn_prev = tk.Button(
            row2, text="<< Prev", command=self._prev,
            font=("monospace", 10, "bold"), bg="#3c3c3c", fg="#e0e0e0",
            activebackground="#505050", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=2, takefocus=False
        )
        self.btn_prev.pack(side=tk.LEFT)

        self.lbl_counter = tk.Label(
            row2, text="", font=("monospace", 11),
            fg="#a0a0a0", bg="#1e1e1e"
        )
        self.lbl_counter.pack(side=tk.LEFT, expand=True)

        self.lbl_file = tk.Label(
            row2, text="", font=("monospace", 9),
            fg="#606060", bg="#1e1e1e"
        )
        self.lbl_file.pack(side=tk.LEFT, expand=True)

        self.btn_next = tk.Button(
            row2, text="Next >>", command=self._next,
            font=("monospace", 10, "bold"), bg="#3c3c3c", fg="#e0e0e0",
            activebackground="#505050", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=2, takefocus=False
        )
        self.btn_next.pack(side=tk.RIGHT)

    def _bind_keys(self):
        self.bind_all("<Left>", lambda _: self._prev())
        self.bind_all("<Right>", lambda _: self._next())
        self.bind_all("<Escape>", lambda _: self._on_close())

    def _update_display(self):
        f = self.rkf_files[self.current_idx]
        self.lbl_molecule.config(text=f["molecule"])
        self.lbl_solvent.config(text=f["solvent"])
        self.lbl_conf.config(text=f"conf {f['conf_num']}")
        self.lbl_calc.config(text=f["calc_type"])
        self.lbl_file.config(text=f["filename"])
        self.lbl_counter.config(
            text=f"{self.current_idx + 1} / {len(self.rkf_files)}"
        )

        self.btn_prev.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.btn_next.config(
            state=tk.NORMAL if self.current_idx < len(self.rkf_files) - 1 else tk.DISABLED
        )

    def _kill_viewer(self):
        if self.viewer_process and self.viewer_process.poll() is None:
            self.viewer_process.terminate()
            try:
                self.viewer_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.viewer_process.kill()
                self.viewer_process.wait()
        self.viewer_process = None

    def _refocus(self):
        """Reprend le focus apres qu'amsview ait ouvert sa fenetre."""
        self.focus_force()
        self.lift()

    def _launch_viewer(self):
        self._kill_viewer()
        rkf_path = self.rkf_files[self.current_idx]["path"]
        try:
            self.viewer_process = subprocess.Popen(
                ["amsview", rkf_path],
                env=self.ams_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Reprendre le focus apres qu'amsview ait eu le temps de s'ouvrir
            self.after(500, self._refocus)
        except FileNotFoundError:
            self.lbl_file.config(
                text="ERREUR: amsview non trouve.",
                fg="#ff6666"
            )

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_display()
            self._launch_viewer()

    def _next(self):
        if self.current_idx < len(self.rkf_files) - 1:
            self.current_idx += 1
            self._update_display()
            self._launch_viewer()

    def _on_close(self):
        self._kill_viewer()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Navigation GUI pour visualiser les fichiers RKF redox avec amsview."
    )
    parser.add_argument(
        "path",
        help="Chemin a scanner (data/, data/Solvant/, data/Solvant/Mol_workdir/, ...)"
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Erreur: {args.path} n'existe pas.")
        sys.exit(1)

    ams_env = get_ams_env()
    if ams_env:
        print("Environnement AMS charge depuis amsbashrc.sh.")
    else:
        print("Environnement AMS detecte ($AMSBIN defini).")

    print(f"Scan des fichiers RKF dans {base_path}...")
    files = scan_rkf_files(base_path)

    if not files:
        print("Aucun fichier RKF trouve dans le chemin specifie.")
        sys.exit(1)

    files = sort_rkf_files(files)
    print(f"{len(files)} fichiers RKF trouves.")

    app = RKFViewerApp(files, ams_env)
    app.mainloop()


if __name__ == "__main__":
    main()
