#!/usr/bin/env python

import os
import sys
import re

def extract_energy_values(out_file):
    """Extrait les valeurs d'énergie et thermodynamiques d'un fichier de sortie AMS/ADF."""
    values = {}
    
    try:
        with open(out_file, 'r') as f:
            content = f.read()
            
            # Chercher l'énergie électronique (appelée "Energy from Engine" dans AMS)
            elec_energy_match = re.search(r"Energy from Engine:\s+([-]?\d+\.\d+)", content)
            if elec_energy_match:
                values['electronic_energy'] = float(elec_energy_match.group(1))
            
            # Chercher l'énergie libre de Gibbs
            gibbs_match = re.search(r"Gibbs free energy:\s+([-]?\d+\.\d+)", content)
            if gibbs_match:
                values['gibbs_free_energy'] = float(gibbs_match.group(1))
            
            # Chercher l'enthalpie
            enthalpy_match = re.search(r"Enthalpy H:\s+([-]?\d+\.\d+)", content)
            if enthalpy_match:
                values['enthalpy'] = float(enthalpy_match.group(1))
            
            # Chercher l'entropie (dans T*S)
            entropy_match = re.search(r"-T\*S:\s+([-]?\d+\.\d+)", content)
            if entropy_match:
                # Notez que nous stockons -T*S, pas S directement
                values['entropy'] = float(entropy_match.group(1))
        
        if not values:
            print(f"AVERTISSEMENT: Aucune valeur d'énergie trouvée dans {out_file}")
            return None
        
        return values
    
    except Exception as e:
        print(f"ERREUR lors de l'extraction des valeurs de {out_file}: {str(e)}")
        return None

def process_conformer(calc_dir, conformer_name, is_neutre, suffix=None):
    if suffix is None:
        suffix = "neutre_opt" if is_neutre else "reduit_opt"
    
    out_file = f"{calc_dir}/{conformer_name}_{suffix}.out"
    
    if not os.path.exists(out_file):
        print(f"ERREUR: Fichier {out_file} introuvable!")
        return None
    
    # Extraction des valeurs d'énergie du fichier de sortie
    energy_values = extract_energy_values(out_file)
    
    if energy_values:
        print(f"Résultats thermodynamiques pour {conformer_name} ({suffix}):")
        print(f"  Énergie électronique: {energy_values['electronic_energy']:.6f} hartree")
        if 'gibbs_free_energy' in energy_values:
            print(f"  Énergie libre de Gibbs: {energy_values['gibbs_free_energy']:.6f} hartree")
        if 'enthalpy' in energy_values:
            print(f"  Enthalpie: {energy_values['enthalpy']:.6f} hartree")
        if 'entropy' in energy_values:
            print(f"  -T*S: {energy_values['entropy']:.6f} hartree")
    
    return energy_values

def calculate_BDE(conformer_name, conformer_neutre_values, conformer_reduit_opt_values, conformer_reduit_sp_values):
    # Utiliser l'énergie électronique du calcul SP du réduit
    E_reduit = conformer_reduit_sp_values['electronic_energy']
    
    # Utiliser l'énergie électronique du neutre
    E_neutre = conformer_neutre_values['electronic_energy']
    
    # Calculer la BDE en hartree
    bde_hartree = E_neutre - E_reduit
    
    # Convertir en kcal/mol (1 hartree = 627.5095 kcal/mol)
    bde_kcal_mol = bde_hartree * 627.5095
    
    print(f"BDE pour {conformer_name}: {bde_kcal_mol:.2f} kcal/mol")
    
    return bde_kcal_mol

def main():
    args = sys.argv[1:]
    
    if not args:
        print("Usage: python script.py chemin_vers_dossier_de_travail")
        sys.exit(1)
    
    work_dir = args[0]
    
    # Trouver tous les conformères uniques
    conformers = set()
    for dir_name in os.listdir(work_dir):
        if "_neutre_opt" in dir_name:
            conformer = dir_name.replace("_neutre_opt", "")
            conformers.add(conformer)
    
    if not conformers:
        print(f"Aucun conformère trouvé dans {work_dir}")
        sys.exit(1)
    
    print(f"Conformères trouvés: {', '.join(sorted(conformers))}")
    print()
    
    bde_values = {}
    
    for conformer in sorted(conformers):
        neutre_dir = f"{work_dir}/{conformer}_neutre_opt"
        reduit_opt_dir = f"{work_dir}/{conformer}_reduit_opt"
        reduit_sp_dir = f"{work_dir}/{conformer}_reduit_sp"
        
        print(f"Traitement de {conformer}...")
        
        # Traitement du conformer neutre
        conformer_neutre_values = process_conformer(neutre_dir, conformer, True, suffix="neutre_opt")
        
        # Traitement du conformer réduit (opt)
        conformer_reduit_opt_values = process_conformer(reduit_opt_dir, conformer, False, suffix="reduit_opt")
        
        # Traitement du conformer réduit (sp)
        conformer_reduit_sp_values = process_conformer(reduit_sp_dir, conformer, False, suffix="reduit_sp")
        
        # Calcul de la BDE si toutes les données sont disponibles
        if conformer_neutre_values and conformer_reduit_opt_values and conformer_reduit_sp_values:
            bde = calculate_BDE(conformer, conformer_neutre_values, conformer_reduit_opt_values, conformer_reduit_sp_values)
            bde_values[conformer] = bde
        else:
            print("AVERTISSEMENT: Impossible de calculer la BDE - valeurs manquantes.")
        
        print()
    
    # Affichage des résultats
    if bde_values:
        print("Résumé des valeurs de BDE:")
        for conformer, bde in sorted(bde_values.items()):
            print(f"{conformer}: {bde:.2f} kcal/mol")
        
        # Identifier le conformer avec la BDE la plus basse
        min_bde_conformer = min(bde_values.items(), key=lambda x: x[1])[0]
        print(f"\nConformère avec la BDE la plus basse: {min_bde_conformer} ({bde_values[min_bde_conformer]:.2f} kcal/mol)")
    else:
        print("Aucune valeur de BDE n'a pu être calculée.")

if __name__ == "__main__":
    main()
