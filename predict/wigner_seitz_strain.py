#!/usr/bin/env python3
"""
Análisis Wigner-Seitz con soporte para STRAIN GRANDE (> 5%)

CARACTERÍSTICAS:
- Detecta vacancias e intersticiales
- Soporta strain > 5% mediante mapeo afín
- Optimizado con NumPy
- Condiciones periódicas de contorno

AUTOR: Basado en OVITO Wigner-Seitz Analysis
"""

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import warnings


# ============================================================================
# CLASES AUXILIARES
# ============================================================================

class SimulationBox:
    """Caja de simulación con métodos para mapeo afín."""
    
    def __init__(self, xlo, xhi, ylo, yhi, zlo, zhi, xy=0.0, xz=0.0, yz=0.0):
        self.xlo, self.xhi = xlo, xhi
        self.ylo, self.yhi = ylo, yhi
        self.zlo, self.zhi = zlo, zhi
        self.xy, self.xz, self.yz = xy, xz, yz
    
    @property
    def lx(self):
        return self.xhi - self.xlo
    
    @property
    def ly(self):
        return self.yhi - self.ylo
    
    @property
    def lz(self):
        return self.zhi - self.zlo
    
    @property
    def dimensions(self):
        """Dimensiones [Lx, Ly, Lz]"""
        return np.array([self.lx, self.ly, self.lz])
    
    @property
    def cell_matrix(self):
        """
        Matriz H: convierte coordenadas fraccionarias -> cartesianas
        Para caja ortogonal: H = diag([Lx, Ly, Lz])
        """
        if abs(self.xy) < 1e-10 and abs(self.xz) < 1e-10 and abs(self.yz) < 1e-10:
            return np.diag([self.lx, self.ly, self.lz])
        
        # Caja triclinica
        return np.array([
            [self.lx, 0.0, 0.0],
            [self.xy, self.ly, 0.0],
            [self.xz, self.yz, self.lz]
        ])
    
    @property
    def reciprocal_cell_matrix(self):
        """Matriz H^-1: convierte cartesianas -> fraccionarias"""
        return np.linalg.inv(self.cell_matrix)
    
    def get_volume(self):
        """Volumen de la celda"""
        return np.abs(np.linalg.det(self.cell_matrix))
    
    def get_strain(self, reference_box):
        """Calcula strain volumétrico: (V - V0) / V0"""
        V0 = reference_box.get_volume()
        V = self.get_volume()
        return (V - V0) / V0
    
    def compute_affine_transformation(self, target_box):
        """
        Calcula transformación afín F que mapea esta celda -> celda objetivo
        F = H_target @ H_current^-1
        """
        return target_box.cell_matrix @ self.reciprocal_cell_matrix
    
    def minimum_image_distance(self, pos1, pos2):
        """Distancia con convención de imagen mínima (PBC)"""
        delta = pos2 - pos1
        dims = self.dimensions
        delta = delta - dims * np.round(delta / dims)
        return delta


# ============================================================================
# LECTURA DE ARCHIVOS LAMMPS
# ============================================================================

def read_lammps_dump(filepath):
    """
    Lee archivo dump de LAMMPS.
    
    Returns:
        (posiciones, SimulationBox)
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ No se encontró: {filepath}")
    
    coords = []
    box_bounds = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if "BOX BOUNDS" in line:
            x_line = lines[i+1].split()
            y_line = lines[i+2].split()
            z_line = lines[i+3].split()
            
            xlo, xhi = float(x_line[0]), float(x_line[1])
            ylo, yhi = float(y_line[0]), float(y_line[1])
            zlo, zhi = float(z_line[0]), float(z_line[1])
            
            xy = float(x_line[2]) if len(x_line) > 2 else 0.0
            xz = float(y_line[2]) if len(y_line) > 2 else 0.0
            yz = float(z_line[2]) if len(z_line) > 2 else 0.0
            
            box_bounds = SimulationBox(xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)
            i += 4
            
        elif "ITEM: ATOMS" in line:
            i += 1
            while i < len(lines):
                parts = lines[i].split()
                if len(parts) >= 5:
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                elif len(parts) == 0:
                    break
                i += 1
            break
        else:
            i += 1
    
    if not coords:
        raise ValueError(f"❌ No se encontraron coordenadas en {filepath}")
    
    return np.array(coords), box_bounds


# ============================================================================
# ANALIZADOR WIGNER-SEITZ
# ============================================================================

class WignerSeitzAnalyzer:
    """
    Analizador de defectos Wigner-Seitz con soporte para strain grande.
    """
    
    def __init__(self, reference_positions, defective_positions, 
                 reference_box, defective_box, 
                 use_pbc=True, use_affine_mapping=False):
        """
        Args:
            reference_positions: Posiciones perfectas (N_sites x 3)
            defective_positions: Posiciones con defectos (N_atoms x 3)
            reference_box: Caja de referencia
            defective_box: Caja defectuosa
            use_pbc: Usar condiciones periódicas
            use_affine_mapping: Activar mapeo afín (USAR SI STRAIN > 5%)
        """
        self.reference = reference_positions.copy()
        self.defective = defective_positions.copy()
        self.reference_box = reference_box
        self.defective_box = defective_box
        self.use_pbc = use_pbc
        self.use_affine_mapping = use_affine_mapping
        
        self.n_sites = len(reference_positions)
        self.n_atoms = len(defective_positions)
        
        # Calcular strain
        self.volumetric_strain = defective_box.get_strain(reference_box)
        
        # Arrays de resultados
        self.occupancy = np.zeros(self.n_sites, dtype=np.int32)
        self.atom_to_site = np.full(self.n_atoms, -1, dtype=np.int64)
        self._site_mapping = None
        
        # Aplicar mapeo afín si está activado
        if self.use_affine_mapping:
            self._apply_affine_mapping()
        
        # Información
        print(f"\n{'='*60}")
        print(f"INICIALIZACIÓN")
        print(f"{'='*60}")
        print(f"Sitios de referencia: {self.n_sites}")
        print(f"Átomos defectuosos: {self.n_atoms}")
        print(f"Strain volumétrico: {self.volumetric_strain*100:.2f}%")
        print(f"Condiciones periódicas: {'Sí' if use_pbc else 'No'}")
        print(f"Mapeo afín: {'✅ ACTIVADO' if use_affine_mapping else '❌ Desactivado'}")
        
        # Warning si strain > 5% y no hay mapeo afín
        if abs(self.volumetric_strain) > 0.05 and not use_affine_mapping:
            print(f"\n{'⚠️ '*20}")
            print(f"⚠️  ADVERTENCIA: Strain > 5% detectado!")
            print(f"⚠️  Sin mapeo afín, los resultados pueden tener errores.")
            print(f"⚠️  Ejecute con: --affine-map")
            print(f"{'⚠️ '*20}")
        print(f"{'='*60}\n")
    
    def _apply_affine_mapping(self):
        """
        Aplica transformación afín para corregir deformación de celda.
        
        ESTO ES CRÍTICO PARA STRAIN > 5%
        """
        print(f"\n{'='*60}")
        print(f"APLICANDO MAPEO AFÍN")
        print(f"{'='*60}")
        
        # F^-1 mapea: configuración defectuosa -> configuración de referencia
        F = self.reference_box.compute_affine_transformation(self.defective_box)
        F_inverse = np.linalg.inv(F)
        
        print(f"Transformación F^-1:")
        print(F_inverse)
        
        # Transformar posiciones
        self.defective = (F_inverse @ self.defective.T).T
        
        print(f"✅ Posiciones remapeadas a espacio de referencia")
        print(f"{'='*60}\n")
    
    def _build_kdtree(self):
        """Construye KDTree con réplicas periódicas si es necesario."""
        if not self.use_pbc:
            self._site_mapping = np.arange(self.n_sites)
            return KDTree(self.reference)
        
        # Crear 27 réplicas (3x3x3 celdas)
        dims = self.reference_box.dimensions
        shifts = np.array([-1, 0, 1])
        
        i_shifts = np.repeat(shifts, 9)
        j_shifts = np.tile(np.repeat(shifts, 3), 3)
        k_shifts = np.tile(shifts, 9)
        
        offsets = np.column_stack([
            i_shifts * dims[0],
            j_shifts * dims[1],
            k_shifts * dims[2]
        ])
        
        # Broadcasting: (n_sites, 1, 3) + (1, 27, 3) = (n_sites, 27, 3)
        expanded = self.reference[:, np.newaxis, :] + offsets[np.newaxis, :, :]
        expanded = expanded.reshape(-1, 3)
        
        self._site_mapping = np.repeat(np.arange(self.n_sites), 27)
        
        return KDTree(expanded)
    
    def assign_atoms_to_sites(self):
        """Asigna cada átomo al sitio más cercano."""
        print("Asignando átomos a sitios...")
        
        tree = self._build_kdtree()
        max_dist = 0.5 * np.min(self.reference_box.dimensions)
        
        distances, indices = tree.query(
            self.defective, 
            k=1,
            distance_upper_bound=max_dist
        )
        
        valid_mask = np.isfinite(distances)
        
        if not np.all(valid_mask):
            n_unassigned = np.sum(~valid_mask)
            warnings.warn(f"{n_unassigned} átomos no asignados")
        
        for atom_idx in np.where(valid_mask)[0]:
            site_idx = self._site_mapping[indices[atom_idx]]
            self.atom_to_site[atom_idx] = site_idx
            self.occupancy[site_idx] += 1
        
        print(f"✅ {np.sum(valid_mask)}/{self.n_atoms} átomos asignados\n")
    
    def find_vacancies(self):
        """Encuentra vacancias (ocupación = 0)."""
        return np.where(self.occupancy == 0)[0]
    
    def find_interstitials(self):
        """Encuentra átomos intersticiales."""
        interstitial_sites = np.where(self.occupancy > 1)[0]
        
        if len(interstitial_sites) == 0:
            return np.array([], dtype=np.int64), interstitial_sites
        
        interstitial_atoms = []
        
        for site_idx in interstitial_sites:
            atoms_at_site = np.where(self.atom_to_site == site_idx)[0]
            
            if len(atoms_at_site) <= 1:
                continue
            
            site_pos = self.reference[site_idx]
            atom_positions = self.defective[atoms_at_site]
            
            if self.use_pbc:
                deltas = self.reference_box.minimum_image_distance(
                    site_pos[np.newaxis, :], atom_positions
                )
                distances = np.linalg.norm(deltas, axis=1)
            else:
                distances = np.linalg.norm(atom_positions - site_pos, axis=1)
            
            # El más cercano es "normal", los demás son intersticiales
            sorted_indices = np.argsort(distances)
            interstitial_atoms.extend(atoms_at_site[sorted_indices[1:]])
        
        return np.array(interstitial_atoms, dtype=np.int64), interstitial_sites
    
    def analyze(self):
        """Ejecuta análisis completo."""
        print(f"{'='*60}")
        print(f"ANÁLISIS WIGNER-SEITZ")
        print(f"{'='*60}\n")
        
        self.assign_atoms_to_sites()
        
        vacancies = self.find_vacancies()
        interstitial_atoms, interstitial_sites = self.find_interstitials()
        
        n_vacancies = len(vacancies)
        n_interstitials = len(interstitial_atoms)
        
        results = {
            'vacancies': vacancies,
            'interstitial_atoms': interstitial_atoms,
            'interstitial_sites': interstitial_sites,
            'occupancy': self.occupancy.copy(),
            'n_vacancies': n_vacancies,
            'n_interstitials': n_interstitials,
            'vacancy_concentration': n_vacancies / self.n_sites,
            'interstitial_concentration': n_interstitials / self.n_atoms,
            'volumetric_strain': self.volumetric_strain,
        }
        
        # Imprimir resultados
        print(f"{'RESULTADOS':^60}")
        print(f"{'-'*60}")
        print(f"Vacancias: {n_vacancies}")
        print(f"  Concentración: {results['vacancy_concentration']:.6f}")
        print(f"\nIntersticiales: {n_interstitials}")
        print(f"  Concentración: {results['interstitial_concentration']:.6f}")
        print(f"\nBalance (átomos - sitios): {self.n_atoms - self.n_sites:+d}")
        print(f"{'='*60}\n")
        
        return results
    
    def plot_defects(self, results, plot_range=20.0, save_path=None):
        """Visualiza defectos en 3D."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sitios normales
        normal = np.where(self.occupancy == 1)[0]
        if len(normal) > 0:
            pos = self.reference[normal]
            mask = np.all(np.abs(pos) < plot_range, axis=1)
            pos = pos[mask]
            if len(pos) > 0:
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                          c='lightblue', s=20, alpha=0.3, label='Normal')
        
        # Vacancias
        if len(results['vacancies']) > 0:
            pos = self.reference[results['vacancies']]
            mask = np.all(np.abs(pos) < plot_range, axis=1)
            pos = pos[mask]
            if len(pos) > 0:
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                          c='red', s=200, marker='x', linewidth=3,
                          label=f'Vacancias ({len(results["vacancies"])})')
        
        # Intersticiales
        if len(results['interstitial_atoms']) > 0:
            pos = self.defective[results['interstitial_atoms']]
            mask = np.all(np.abs(pos) < plot_range, axis=1)
            pos = pos[mask]
            if len(pos) > 0:
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                          c='orange', s=200, marker='*', linewidth=2,
                          label=f'Intersticiales ({len(results["interstitial_atoms"])})')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_zlim(-plot_range, plot_range)
        
        title = 'Análisis Wigner-Seitz'
        if self.use_affine_mapping:
            title += ' (con mapeo afín)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figura guardada: {save_path}")
        
        plt.show()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal."""
    if len(sys.argv) < 3:
        print("""
╔════════════════════════════════════════════════════════════════╗
║        ANÁLISIS WIGNER-SEITZ CON SOPORTE STRAIN > 5%          ║
╚════════════════════════════════════════════════════════════════╝

USO:
    python wigner_seitz_strain.py <referencia.dump> <defectuosa.dump> [opciones]

OPCIONES:
    --no-pbc            Desactivar condiciones periódicas
    --affine-map        Activar mapeo afín (USE PARA STRAIN > 5%)
    --plot-range N      Rango de visualización (default: 15)
    --save ARCHIVO      Guardar figura PNG

EJEMPLOS:

    1. Análisis básico:
       python wigner_seitz_strain.py ref.dump def.dump

    2. Con strain > 5% (IMPORTANTE):
       python wigner_seitz_strain.py ref.dump def.dump --affine-map

    3. Completo:
       python wigner_seitz_strain.py ref.dump def.dump \\
           --affine-map --plot-range 25 --save resultados.png

⚠️  IMPORTANTE: Si tiene strain > 5%, DEBE usar --affine-map
        """)
        return 1
    
    ref_file = sys.argv[1]
    def_file = sys.argv[2]
    
    # Parsear opciones
    use_pbc = '--no-pbc' not in sys.argv
    use_affine = '--affine-map' in sys.argv
    
    plot_range = 15.0
    if '--plot-range' in sys.argv:
        idx = sys.argv.index('--plot-range')
        plot_range = float(sys.argv[idx + 1])
    
    save_path = None
    if '--save' in sys.argv:
        idx = sys.argv.index('--save')
        save_path = sys.argv[idx + 1]
    
    try:
        # Cargar archivos
        print(f"\n📂 Cargando: {ref_file}")
        ref_positions, ref_box = read_lammps_dump(ref_file)
        print(f"   ✅ {len(ref_positions)} sitios")
        
        print(f"\n📂 Cargando: {def_file}")
        def_positions, def_box = read_lammps_dump(def_file)
        print(f"   ✅ {len(def_positions)} átomos")
        
        # Crear analizador
        analyzer = WignerSeitzAnalyzer(
            ref_positions, def_positions,
            ref_box, def_box,
            use_pbc=use_pbc,
            use_affine_mapping=use_affine
        )
        
        # Analizar
        results = analyzer.analyze()
        
        # Visualizar
        print("Generando visualización...")
        analyzer.plot_defects(results, plot_range=plot_range, save_path=save_path)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())