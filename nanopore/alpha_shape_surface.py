import numpy as np
from scipy.spatial import Delaunay, distance_matrix
from scipy.spatial.distance import cdist
import warnings

class AlphaShapeSurfaceConstructor:
    """
    Constructs surface mesh using Alpha Shape method (like OVITO's ConstructSurfaceModifier).
    Detects pores/voids by testing if a probe sphere of given radius can fit in space.
    """
    
    def __init__(self, positions, probe_radius, smoothing_level=0, 
                 pbc_flags=None, select_surface_particles=False):
        """
        Args:
            positions: Nx3 array of atom coordinates
            probe_radius: radius of probe sphere for filtering
            smoothing_level: number of smoothing iterations (Laplacian smoothing)
            pbc_flags: tuple of 3 bools for periodic boundary conditions (x, y, z)
            select_surface_particles: mark atoms on surface
        """
        self.positions = np.array(positions, dtype=np.float64)
        self.probe_radius = probe_radius
        self.alpha = probe_radius * probe_radius  # Alpha parameter for Delaunay filtering
        self.smoothing_level = smoothing_level
        self.pbc_flags = pbc_flags or (False, False, False)
        self.select_surface_particles = select_surface_particles
        
        self.surface_vertices = None
        self.surface_faces = None
        self.surface_particle_selection = None
        self.surface_area = None
    def get_surface_atoms_indices(self):
        """Retorna los índices de átomos en la superficie"""
        if not hasattr(self, '_surface_atom_indices'):
            if self.surface_particle_selection is not None:
                return np.where(self.surface_particle_selection)[0]
            return np.array([], dtype=int)
        return self._surface_atom_indices

    def filter_surface_atoms_exclude_box_boundary(self, box_bounds=None, margin=0.01):
        """
        Filtra átomos superficiales para excluir los que están en el borde de la caja.
        
        Args:
            box_bounds: tupla ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                    Si es None, se calcula automáticamente
            margin: distancia en unidades desde el borde (átomos más cercanos se excluyen)
        
        Returns:
            np.array: índices de átomos superficiales que NO están en el borde
        """
        surface_atoms = self.get_surface_atoms_indices()
        
        if len(surface_atoms) == 0:
            return np.array([], dtype=int)
        
        # Auto-detectar límites de caja si no se proporcionan
        if box_bounds is None:
            box_bounds = (
                (self.positions[:, 0].min(), self.positions[:, 0].max()),
                (self.positions[:, 1].min(), self.positions[:, 1].max()),
                (self.positions[:, 2].min(), self.positions[:, 2].max())
            )
        
        interior_atoms = []
        
        for atom_idx in surface_atoms:
            pos = self.positions[atom_idx]
            on_boundary = False
            
            # Verificar distancia a cada cara de la caja
            for dim in range(3):
                dist_to_min = pos[dim] - box_bounds[dim][0]
                dist_to_max = box_bounds[dim][1] - pos[dim]
                
                # Si está más cerca del borde que 'margin', está en borde
                if dist_to_min < margin or dist_to_max < margin:
                    on_boundary = True
                    break
            
            if not on_boundary:
                interior_atoms.append(atom_idx)
        
        return np.array(interior_atoms, dtype=int)
        
    def perform(self):
        """Main algorithm: compute Delaunay, filter by alpha, construct surface"""
        
        if self.probe_radius <= 0:
            raise ValueError("Probe radius must be positive")
        
        # Step 1: Generate Delaunay tessellation
        print("Generating Delaunay tessellation...")
        try:
            delaunay = Delaunay(self.positions)
        except Exception as e:
            raise RuntimeError(f"Failed to generate Delaunay tessellation: {e}")
        
        # Step 2: Filter tetrahedra based on alpha parameter
        print("Filtering tetrahedra by alpha parameter...")
        valid_tets = self._filter_tetrahedra(delaunay)
        print(f"Valid tetrahedra: {len(valid_tets)} / {len(delaunay.simplices)}")
        
        # Step 3: Extract surface facets from valid tetrahedra
        print("Extracting surface facets...")
        surface_facets = self._extract_surface_facets(delaunay, valid_tets)
        
        # Step 4: Build surface mesh from facets
        print("Building surface mesh...")
        self.surface_vertices, self.surface_faces = self._build_mesh(
            delaunay, surface_facets, valid_tets
        )
        
        # Step 5: Mark surface particles
        if self.select_surface_particles:
            self.surface_particle_selection = self._select_surface_particles(delaunay, surface_facets)
        
        # Step 6: Smooth mesh
        if self.smoothing_level > 0:
            print(f"Smoothing mesh ({self.smoothing_level} iterations)...")
            self.surface_vertices = self._smooth_mesh(
                self.surface_vertices, self.surface_faces, self.smoothing_level
            )
        
        # Step 7: Compute surface area
        self.surface_area = self._compute_surface_area()
        print(f"Surface area: {self.surface_area:.4f}")
        
        return self
    
    def _filter_tetrahedra(self, delaunay):
        """
        Filter tetrahedra based on circumradius.
        A tetrahedron is valid if its circumradius <= probe_radius
        (i.e., alpha filtering: keeps filled regions)
        """
        valid_tets = []
        
        for tet_idx, tet in enumerate(delaunay.simplices):
            # Get the 4 vertices of the tetrahedron
            verts = self.positions[tet]
            
            # Compute circumcenter and circumradius
            circumradius = self._compute_circumradius(verts)
            
            # Keep tetrahedron if circumradius <= probe_radius
            if circumradius <= self.probe_radius:
                valid_tets.append(tet_idx)
        
        return np.array(valid_tets)
    
    def _compute_circumradius(self, vertices):
        """
        Compute circumradius of a tetrahedron given 4 vertices (Nx3).
        Circumradius R is the radius of the sphere passing through all 4 vertices.
        Uses Cayley-Menger determinant method.
        """
        v0, v1, v2, v3 = vertices
        
        # Compute all 6 edge lengths
        d01 = np.linalg.norm(v1 - v0)
        d02 = np.linalg.norm(v2 - v0)
        d03 = np.linalg.norm(v3 - v0)
        d12 = np.linalg.norm(v2 - v1)
        d13 = np.linalg.norm(v3 - v1)
        d23 = np.linalg.norm(v3 - v2)
        
        # Volume using scalar triple product
        a = v1 - v0
        b = v2 - v0
        c = v3 - v0
        volume = abs(np.dot(a, np.cross(b, c))) / 6.0
        
        if volume < 1e-12:  # Degenerate tetrahedron
            return np.inf
        
        # Cayley-Menger determinant approach for circumradius
        # R = product_of_edges / (288 * volume^2) [alternative formula]
        # But more reliable: solve the circumsphere directly
        
        # System: ||center - v_i||^2 = R^2 for all 4 vertices
        # Build linear system: A*center = b
        A = np.array([
            2*(v1 - v0),
            2*(v2 - v0),
            2*(v3 - v0)
        ])
        
        b = np.array([
            np.dot(v1, v1) - np.dot(v0, v0),
            np.dot(v2, v2) - np.dot(v0, v0),
            np.dot(v3, v3) - np.dot(v0, v0)
        ])
        
        try:
            center = np.linalg.solve(A, b)
            R = np.linalg.norm(center - v0)
            return R
        except np.linalg.LinAlgError:
            return np.inf
    
    def _extract_surface_facets(self, delaunay, valid_tets):
        """
        Extract surface facets correctly:
        A facet is on the surface if it borders a valid tetrahedron and an INVALID 
        (or non-existent) tetrahedron.
        """
        valid_tet_set = set(valid_tets)
        facet_to_tets = {}  # Map facet to list of (tet_idx, is_valid)
        
        # Build complete adjacency for all tetrahedra
        for tet_idx, tet in enumerate(delaunay.simplices):
            is_valid = tet_idx in valid_tet_set
            
            # 4 facets per tetrahedron (opposite to each vertex)
            for i in range(4):
                facet = tuple(sorted(np.delete(tet, i)))
                facet_key = frozenset(facet)
                
                if facet_key not in facet_to_tets:
                    facet_to_tets[facet_key] = []
                facet_to_tets[facet_key].append((tet_idx, is_valid))
        
        # Surface facets: on boundary between valid and invalid (or only valid)
        surface_facets = []
        for facet_key, tet_list in facet_to_tets.items():
            # Count valid tetrahedra sharing this facet
            valid_count = sum(1 for _, is_valid in tet_list if is_valid)
            
            # Surface facet if:
            # - Appears in exactly 1 valid tet (and possibly invalid ones), OR
            # - Appears in 1 valid tet and 0 invalid tets (boundary of point cloud)
            if valid_count == 1:
                surface_facets.append(list(facet_key))
        
        return surface_facets
    
    def _build_mesh(self, delaunay, surface_facets, valid_tets):
        """
        Build surface mesh from facets.
        Returns vertices and face indices, keeping track of original particle indices.
        """
        if not surface_facets:
            self._surface_atom_indices = np.array([], dtype=int)
            return np.array([]), np.array([])
        
        # Get unique vertex indices from surface facets
        surface_vertex_indices = sorted(set(np.array(surface_facets).flatten()))
        
        # Store ORIGINAL indices (from input positions) - these are the atoms we keep
        self._surface_atom_indices = np.array(surface_vertex_indices, dtype=int)
        
        # Create mapping from original indices to new mesh vertex indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(surface_vertex_indices)}
        
        # Extract surface vertices from original positions
        vertices = self.positions[surface_vertex_indices]
        
        # Remap face indices to new vertex numbering
        faces = []
        for facet in surface_facets:
            remapped_facet = [vertex_map[v] for v in facet]
            faces.append(remapped_facet)
        
        return vertices, np.array(faces)
    
    def _select_surface_particles(self, delaunay, surface_facets):
        """Mark input particles that are on the surface"""
        selection = np.zeros(len(self.positions), dtype=np.int32)
        
        for facet in surface_facets:
            for particle_idx in facet:
                selection[particle_idx] = 1
        
        return selection
    
    def _smooth_mesh(self, vertices, faces, iterations):
        """
        Laplacian smoothing of mesh vertices.
        Only smooth interior vertices, preserve boundary vertices.
        """
        smoothed_vertices = vertices.copy()
        
        # Identify boundary vertices (appear in only one face or form open boundary)
        vertex_degree = {}
        for face in faces:
            for v_idx in face:
                vertex_degree[v_idx] = vertex_degree.get(v_idx, 0) + 1
        
        for iteration in range(iterations):
            new_vertices = smoothed_vertices.copy()
            
            # Build adjacency: which vertices are neighbors of each vertex
            vertex_neighbors = [set() for _ in range(len(smoothed_vertices))]
            for face in faces:
                for i in range(len(face)):
                    for j in range(len(face)):
                        if i != j:
                            vertex_neighbors[face[i]].add(face[j])
            
            # Update each interior vertex (not boundary)
            for v_idx in range(len(smoothed_vertices)):
                neighbors = list(vertex_neighbors[v_idx])
                if neighbors and len(neighbors) > 2:  # Interior vertex
                    new_vertices[v_idx] = smoothed_vertices[neighbors].mean(axis=0)
            
            smoothed_vertices = new_vertices
        
        return smoothed_vertices
    
    def _compute_surface_area(self):
        """Compute total surface area of mesh"""
        if self.surface_faces is None or len(self.surface_faces) == 0:
            return 0.0
        
        total_area = 0.0
        for face in self.surface_faces:
            # Triangle area using cross product
            if len(face) == 3:
                v0, v1, v2 = self.surface_vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                total_area += area
        
        return total_area
    
    def filter_surface_atoms_exclude_box_boundary(self, box_bounds=None, margin=0.01):
        """
        Filter surface atoms to exclude those on the box boundary.
        
        Args:
            box_bounds: tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                       If None, computes from positions
            margin: tolerance distance from box edge (atoms closer than this are excluded)
        
        Returns:
            np.array: indices of surface atoms that are NOT on the box boundary
        """
        if not hasattr(self, '_surface_atom_indices') or len(self._surface_atom_indices) == 0:
            return np.array([], dtype=int)
        
        if box_bounds is None:
            # Auto-detect box bounds from all positions
            box_bounds = (
                (self.positions[:, 0].min(), self.positions[:, 0].max()),
                (self.positions[:, 1].min(), self.positions[:, 1].max()),
                (self.positions[:, 2].min(), self.positions[:, 2].max())
            )
        
        interior_atoms = []
        for atom_idx in self._surface_atom_indices:
            pos = self.positions[atom_idx]
            
            # Check if atom is inside the box (not on boundary)
            on_boundary = False
            for dim in range(3):
                dist_to_min = pos[dim] - box_bounds[dim][0]
                dist_to_max = box_bounds[dim][1] - pos[dim]
                
                if dist_to_min < margin or dist_to_max < margin:
                    on_boundary = True
                    break
            
            if not on_boundary:
                interior_atoms.append(atom_idx)
        
        return np.array(interior_atoms, dtype=int)
    
    def get_filtered_surface_atoms(self, exclude_box_boundary=True, box_bounds=None, margin=0.01):
        """
        Get final filtered surface atoms (atoms around nanopores).
        
        Args:
            exclude_box_boundary: if True, removes atoms on box edges
            box_bounds: box limits for filtering
            margin: tolerance for box boundary
        
        Returns:
            np.array: indices of filtered surface atoms
        """
        if exclude_box_boundary:
            return self.filter_surface_atoms_exclude_box_boundary(box_bounds, margin)
        else:
            return self.get_surface_atoms_indices()
    
    def export_filtered_atoms_dump(self, output_filename, atom_data=None, 
                                   exclude_box_boundary=True, box_bounds=None, margin=0.01):
        """
        Export filtered surface atoms to LAMMPS dump format.
        
        Args:
            output_filename: path to save dump file
            atom_data: original atom data dict with 'id', 'type', 'x', 'y', 'z', and other properties
            exclude_box_boundary: if True, removes atoms on box edges
            box_bounds: box limits
            margin: tolerance for box boundary
        """
        filtered_indices = self.get_filtered_surface_atoms(
            exclude_box_boundary, box_bounds, margin
        )
        
        if len(filtered_indices) == 0:
            print("Warning: No surface atoms to export!")
            return
        
        with open(output_filename, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(filtered_indices)}\n")
            
            # Determine box bounds if not provided
            if box_bounds is None:
                box_bounds = (
                    (self.positions[:, 0].min(), self.positions[:, 0].max()),
                    (self.positions[:, 1].min(), self.positions[:, 1].max()),
                    (self.positions[:, 2].min(), self.positions[:, 2].max())
                )
            
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{box_bounds[0][0]:.6f} {box_bounds[0][1]:.6f}\n")
            f.write(f"{box_bounds[1][0]:.6f} {box_bounds[1][1]:.6f}\n")
            f.write(f"{box_bounds[2][0]:.6f} {box_bounds[2][1]:.6f}\n")
            
            # Write atom data
            if atom_data is not None:
                header_cols = atom_data.get('columns', ['id', 'type', 'x', 'y', 'z'])
                f.write(f"ITEM: ATOMS {' '.join(header_cols)}\n")
                
                for new_idx, orig_idx in enumerate(filtered_indices):
                    row_data = []
                    for col in header_cols:
                        if col == 'id':
                            row_data.append(str(orig_idx + 1))  # LAMMPS uses 1-based indexing
                        elif col in atom_data:
                            row_data.append(str(atom_data[col][orig_idx]))
                        else:
                            row_data.append("0")
                    f.write(" ".join(row_data) + "\n")
            else:
                # Simple format without atom properties
                f.write("ITEM: ATOMS id type x y z\n")
                for orig_idx in filtered_indices:
                    atom_id = orig_idx + 1
                    atom_type = 1
                    x, y, z = self.positions[orig_idx]
                    f.write(f"{atom_id} {atom_type} {x:.8f} {y:.8f} {z:.8f}\n")
        
        print(f"Exported {len(filtered_indices)} surface atoms to {output_filename}")
    
    @staticmethod
    def _point_to_triangle_distance(p, a, b, c):
        """
        Minimum distance from point p to triangle (a, b, c).
        Uses barycentric coordinates with proper clamping.
        """
        ab = b - a
        ac = c - a
        ap = p - a
        
        ab_dot_ab = np.dot(ab, ab)
        ab_dot_ac = np.dot(ab, ac)
        ac_dot_ac = np.dot(ac, ac)
        ap_dot_ab = np.dot(ap, ab)
        ap_dot_ac = np.dot(ap, ac)
        
        denom = ab_dot_ab * ac_dot_ac - ab_dot_ac * ab_dot_ac
        if abs(denom) < 1e-10:
            # Degenerate triangle, return distance to closest vertex
            return min(
                np.linalg.norm(p - a),
                np.linalg.norm(p - b),
                np.linalg.norm(p - c)
            )
        
        u = (ac_dot_ac * ap_dot_ab - ab_dot_ac * ap_dot_ac) / denom
        v = (ab_dot_ab * ap_dot_ac - ab_dot_ac * ap_dot_ab) / denom
        w = 1.0 - u - v
        
        # Clamp to triangle domain
        if u < 0:
            u = 0.0
        if v < 0:
            v = 0.0
        if w < 0:
            w = 0.0
        
        # Normalize to keep inside triangle
        total = u + v + w
        if total > 1e-10:
            u /= total
            v /= total
            w /= total
        
        closest_point = u * a + v * b + w * c
        return np.linalg.norm(p - closest_point)


    # Example usage
if __name__ == "__main__":
    # Load example data from dump file
    def load_dump_file(filename):
        """Simple LAMMPS dump file parser"""
        positions = []
        atom_ids = []
        atom_types = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        in_atoms = False
        for i, line in enumerate(lines):
            if "ATOMS" in line and "NUMBER" in line:
                n_atoms = int(lines[i+1].strip())
            elif line.startswith("ITEM: ATOMS"):
                in_atoms = True
                continue
            elif in_atoms and line.startswith("ITEM:"):
                break
            elif in_atoms:
                parts = line.split()
                if len(parts) >= 5:
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    positions.append([x, y, z])
                    atom_ids.append(atom_id)
                    atom_types.append(atom_type)
        
        return np.array(positions), np.array(atom_ids), np.array(atom_types)
    
    # Example: Create synthetic FCC-like structure with pores
    np.random.seed(42)
    positions = np.random.rand(50, 3) * 10
    

    # Opción A: Si usas tu código actual (ConstructSurfaceMesh.py)
    constructor = AlphaShapeSurfaceConstructor(
        positions=positions,
        probe_radius=2.0,
        smoothing_level=1,
        select_surface_particles=True
    )

    constructor.perform()

    # NUEVO: Filtrar átomos de borde
    surface_atoms_cleaned = constructor.filter_surface_atoms_exclude_box_boundary(
        box_bounds=None,  # Auto-detecta de los datos
        margin=0.1        # Ajusta según tamaño de caja (en unidades de simulación)
    )

    print(f"Átomos superficiales (con borde): {np.sum(constructor.surface_particle_selection)}")
    print(f"Átomos superficiales (sin borde): {len(surface_atoms_cleaned)}")


    print(f"\nResults:")
    print(f"Total atoms: {len(positions)}")
    print(f"Surface mesh vertices: {len(constructor.surface_vertices)}")
    print(f"Surface mesh faces: {len(constructor.surface_faces)}")
    print(f"Surface area: {constructor.surface_area:.4f}")
    print(f"Surface atoms (nanopore walls): {len(surface_atoms_idx)}")
    print(f"Surface atom indices: {surface_atoms_idx}")
    
    # Export filtered atoms to new dump file
    atom_data = {
        'id': np.arange(1, len(positions) + 1),
        'type': np.ones(len(positions), dtype=int),
        'columns': ['id', 'type', 'x', 'y', 'z']
    }
    # constructor.export_filtered_atoms_dump('nanopore_atoms.dump', atom_data)