"""
Script mejorado: Alpha Shape + Post-procesamiento para eliminar borde de caja

Uso:
    python process_dump.py input.dump output.dump --probe-radius 2.0 --smoothing 8 --boundary-cutoff 2.0
"""

import numpy as np
import argparse
from collections import OrderedDict


class LAMMPSDumpParser:
    """Parser robusto para archivos LAMMPS dump"""
    
    @staticmethod
    def read(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        data = {
            'timestep': 0,
            'n_atoms': 0,
            'box_bounds': None,
            'columns': [],
            'atoms': OrderedDict(),
            'positions': None,
            'atom_ids_ordered': []
        }
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "ITEM: TIMESTEP":
                data['timestep'] = int(lines[i+1].strip())
                i += 2
            
            elif line == "ITEM: NUMBER OF ATOMS":
                data['n_atoms'] = int(lines[i+1].strip())
                i += 2
            
            elif line.startswith("ITEM: BOX BOUNDS"):
                parts = line.split()
                pbc = parts[3:] if len(parts) > 3 else ['pp', 'pp', 'pp']
                
                bounds = []
                i += 1
                for j in range(3):
                    if i < len(lines):
                        bound_line = lines[i].strip().split()
                        if bound_line and not bound_line[0].startswith("ITEM:"):
                            bounds.append((float(bound_line[0]), float(bound_line[1])))
                            i += 1
                        else:
                            break
                
                data['box_bounds'] = tuple(bounds)
                data['pbc'] = pbc
                continue
            
            elif line.startswith("ITEM: ATOMS"):
                parts = line.split()
                data['columns'] = parts[2:]
                
                positions_list = []
                i += 1
                
                while i < len(lines) and not lines[i].startswith("ITEM:"):
                    atom_line = lines[i].strip()
                    if atom_line:
                        values = atom_line.split()
                        
                        atom_dict = {}
                        for col_idx, col_name in enumerate(data['columns']):
                            try:
                                val = float(values[col_idx])
                                if col_name == 'id' or col_name == 'type':
                                    val = int(val)
                            except (ValueError, IndexError):
                                val = values[col_idx]
                            atom_dict[col_name] = val
                        
                        atom_id = int(values[0])
                        data['atoms'][atom_id] = atom_dict
                        data['atom_ids_ordered'].append(atom_id)
                        
                        if 'x' in data['columns']:
                            x_idx = data['columns'].index('x')
                            y_idx = data['columns'].index('y')
                            z_idx = data['columns'].index('z')
                            positions_list.append([
                                float(values[x_idx]),
                                float(values[y_idx]),
                                float(values[z_idx])
                            ])
                    i += 1
                
                data['positions'] = np.array(positions_list)
                continue
            
            i += 1
        
        return data
    
    @staticmethod
    def write(filename, data, filtered_atom_ids):
        with open(filename, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{data['timestep']}\n")
            
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(filtered_atom_ids)}\n")
            
            f.write("ITEM: BOX BOUNDS")
            if 'pbc' in data:
                f.write(f" {' '.join(data['pbc'])}\n")
            else:
                f.write(" pp pp pp\n")
            
            for xmin, xmax in data['box_bounds']:
                f.write(f"{xmin:.6f} {xmax:.6f}\n")
            
            f.write(f"ITEM: ATOMS {' '.join(data['columns'])}\n")
            
            for atom_id in filtered_atom_ids:
                if atom_id in data['atoms']:
                    atom = data['atoms'][atom_id]
                    values = []
                    for col in data['columns']:
                        val = atom[col]
                        if isinstance(val, (int, float)):
                            if isinstance(val, float):
                                values.append(f"{val:.8f}")
                            else:
                                values.append(str(int(val)))
                        else:
                            values.append(str(val))
                    f.write(" ".join(values) + "\n")


def remove_boundary_surface_atoms(surface_atom_indices, positions, box_bounds, cutoff_distance):
    """
    Post-procesamiento: elimina átomos superficiales que están en el borde de la caja.
    
    Args:
        surface_atom_indices: índices de átomos en la superficie (de Alpha Shape)
        positions: array Nx3 de posiciones
        box_bounds: límites ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        cutoff_distance: distancia mínima del borde (Å)
    
    Returns:
        filtered_indices: índices de átomos que están lejos del borde
    """
    filtered = []
    removed_by_dimension = {'x': 0, 'y': 0, 'z': 0}
    
    for atom_idx in surface_atom_indices:
        pos = positions[atom_idx]
        
        # Distancias a todas las 6 caras
        dist_x_min = pos[0] - box_bounds[0][0]
        dist_x_max = box_bounds[0][1] - pos[0]
        dist_y_min = pos[1] - box_bounds[1][0]
        dist_y_max = box_bounds[1][1] - pos[1]
        dist_z_min = pos[2] - box_bounds[2][0]
        dist_z_max = box_bounds[2][1] - pos[2]
        
        # Distancia mínima a cualquier cara
        distances = {
            'x_min': dist_x_min, 'x_max': dist_x_max,
            'y_min': dist_y_min, 'y_max': dist_y_max,
            'z_min': dist_z_min, 'z_max': dist_z_max
        }
        
        min_dist = min(distances.values())
        min_face = min(distances, key=distances.get)
        
        # Si está suficientemente lejos del borde, lo mantenemos
        if min_dist >= cutoff_distance:
            filtered.append(atom_idx)
        else:
            dim = min_face.split('_')[0]
            removed_by_dimension[dim] += 1
    
    return np.array(filtered, dtype=int), removed_by_dimension


def main():
    parser = argparse.ArgumentParser(
        description="Filtrar átomos alrededor de defectos (Alpha Shape + Post-procesamiento)"
    )
    parser.add_argument("input_dump", help="Archivo dump de entrada")
    parser.add_argument("output_dump", help="Archivo dump de salida")
    parser.add_argument("--probe-radius", type=float, default=2.0,
                        help="Radio de la sonda")
    parser.add_argument("--smoothing", type=int, default=8,
                        help="Nivel de suavizado de la malla")
    parser.add_argument("--boundary-cutoff", type=float, default=2.0,
                        help="Distancia mínima del borde (Å) - post-procesamiento")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Alpha Shape + Post-procesamiento")
    print(f"{'='*80}\n")
    
    # [1] Leer archivo
    print(f"[1/6] Leyendo archivo: {args.input_dump}")
    data = LAMMPSDumpParser.read(args.input_dump)
    
    print(f"      ✓ Átomos totales: {len(data['atoms'])}")
    print(f"      ✓ Posiciones: {data['positions'].shape}")
    print(f"      ✓ Caja: {data['box_bounds']}\n")
    
    # [2] Importar Alpha Shape
    try:
        from alpha_shape_surface import AlphaShapeSurfaceConstructor
    except ImportError:
        print("❌ ERROR: No se pudo importar AlphaShapeSurfaceConstructor")
        print("   Asegúrate de que alpha_shape_surface.py esté en el mismo directorio")
        return
    
    # [3] Ejecutar Alpha Shape
    print(f"[2/6] Ejecutando Alpha Shape...")
    print(f"      Probe radius: {args.probe_radius}")
    print(f"      Smoothing: {args.smoothing}\n")
    
    try:
        constructor = AlphaShapeSurfaceConstructor(
            positions=data['positions'],
            probe_radius=args.probe_radius,
            smoothing_level=args.smoothing,
            select_surface_particles=True
        )
        constructor.perform()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    surface_atoms = constructor.get_surface_atoms_indices()
    print(f"[3/6] Resultados Alpha Shape:")
    print(f"      ✓ Átomos superficiales: {len(surface_atoms)}")
    print(f"      ✓ Vértices de malla: {len(constructor.surface_vertices)}")
    print(f"      ✓ Caras: {len(constructor.surface_faces)}")
    print(f"      ✓ Área: {constructor.surface_area:.4f}\n")
    
    # [4] Post-procesamiento: eliminar borde
    print(f"[4/6] Post-procesamiento: Eliminando átomos del borde...")
    print(f"      Cutoff distance: {args.boundary_cutoff} Å\n")
    
    try:
        filtered_indices, removed_by_dim = remove_boundary_surface_atoms(
            surface_atoms,
            data['positions'],
            data['box_bounds'],
            args.boundary_cutoff
        )
        
        removed_boundary = len(surface_atoms) - len(filtered_indices)
        
        print(f"      ✓ Átomos removidos (borde): {removed_boundary}")
        print(f"        - Dimensión X: {removed_by_dim['x']}")
        print(f"        - Dimensión Y: {removed_by_dim['y']}")
        print(f"        - Dimensión Z: {removed_by_dim['z']}")
        print(f"      ✓ Átomos restantes (defectos): {len(filtered_indices)}\n")
        
    except Exception as e:
        print(f"❌ ERROR en post-procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # [5] Convertir índices a IDs
    print(f"[5/6] Preparando salida...")
    
    filtered_atom_ids = sorted([
        data['atom_ids_ordered'][idx] for idx in filtered_indices
    ])
    
    print(f"      ✓ {len(filtered_atom_ids)} átomos listos para exportar\n")
    
    # [6] Escribir archivo
    print(f"[6/6] Escribiendo: {args.output_dump}")
    
    try:
        LAMMPSDumpParser.write(args.output_dump, data, filtered_atom_ids)
        
        print(f"\n{'='*80}")
        print(f"✓ Completado exitosamente")
        print(f"{'='*80}\n")
        
        print(f"Resumen:")
        print(f"  Entrada:  {len(data['atoms'])} átomos")
        print(f"  Salida:   {len(filtered_atom_ids)} átomos")
        print(f"  Removidos: {len(data['atoms']) - len(filtered_atom_ids)} ({100*(len(data['atoms']) - len(filtered_atom_ids))/len(data['atoms']):.1f}%)\n")
        
    except Exception as e:
        print(f"❌ ERROR escribiendo archivo: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()