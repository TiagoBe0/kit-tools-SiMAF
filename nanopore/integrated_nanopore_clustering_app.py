import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback
import shutil
import tempfile
import json
from pathlib import Path
from collections import OrderedDict
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any

from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from scipy.spatial import Delaunay
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Nanoporos → Clustering",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Pipeline: Nanoporos + Clustering Jerárquico")
st.markdown("**Flujo completo**: Filtrado de nanoporos → Clustering → Exportación")
st.markdown("---")

# ==========================================
# INICIALIZAR SESSION STATE
# ==========================================
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'step': 1,
        'original_data': None,
        'alpha_done': False,
        'filtered_data': None,
        'clustering_done': False,
        'clustering_result': None,
    }

# ==========================================
# PARSERS PARA LAMMPS DUMP
# ==========================================

@st.cache_data
def parse_lammps_dump(file_content: bytes) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Lee archivo LAMMPS dump con manejo robusto de errores"""
    lines = file_content.decode('utf-8').split('\n')
    
    header = {'box_bounds': []}
    atom_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == "ITEM: TIMESTEP":
            header['timestep'] = int(lines[i+1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            header['n_atoms_header'] = int(lines[i+1].strip())
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            i += 1
            for _ in range(3):
                if i < len(lines):
                    bound_line = lines[i].strip()
                    bound_line = bound_line.replace('0.00.0', '0.0 0.0').replace('105.60.0', '105.6 0.0')
                    parts = [float(x) for x in bound_line.split()]
                    if parts:
                        lo, hi = min(parts), max(parts)
                        header['box_bounds'].append([lo, hi])
                    else:
                        header['box_bounds'].append([0.0, 0.0])
                    i += 1
        elif line.startswith("ITEM: ATOMS"):
            columns = line.split()[2:]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("ITEM:"):
                atom_line = lines[i].strip()
                parts = atom_line.split()
                if len(parts) == len(columns):
                    atom_lines.append(atom_line + '\n')
                i += 1
        else:
            i += 1
    
    if not atom_lines:
        raise ValueError("No se encontraron líneas de datos válidas")
    
    data_io = StringIO("".join(atom_lines))
    df = pd.read_csv(data_io, sep=r'\s+', names=columns)
    
    return header, df


def write_lammps_dump(output_path: str, header: Dict[str, Any], df: pd.DataFrame):
    """Escribe archivo LAMMPS dump"""
    with open(output_path, 'w') as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{header['timestep']}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(df)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for bounds in header['box_bounds']:
            f.write(f"{bounds[0]} {bounds[1]}\n")
        f.write(f"ITEM: ATOMS {' '.join(df.columns)}\n")
        f.write(df.to_string(header=False, index=False, float_format="%.8f"))
        f.write("\n")


# ==========================================
# MÓDULO 1: ALPHA SHAPE
# ==========================================

class AlphaShapeSurfaceConstructor:
    """Construye superficie usando Alpha Shape"""
    
    def __init__(self, positions, probe_radius, smoothing_level=0):
        self.positions = np.array(positions, dtype=np.float64)
        self.probe_radius = probe_radius
        self.smoothing_level = smoothing_level
        self.surface_vertices = None
        self.surface_faces = None
        self._surface_atom_indices = None
        self.surface_area = None
    
    def perform(self):
        """Algoritmo principal"""
        if self.probe_radius <= 0:
            raise ValueError("Probe radius debe ser positivo")
        
        delaunay = Delaunay(self.positions)
        valid_tets = self._filter_tetrahedra(delaunay)
        surface_facets = self._extract_surface_facets(delaunay, valid_tets)
        self.surface_vertices, self.surface_faces = self._build_mesh(delaunay, surface_facets)
        self.surface_area = self._compute_surface_area()
        
        return self
    
    def _filter_tetrahedra(self, delaunay):
        """Filtra tetraedros por circumradius"""
        valid_tets = []
        for tet_idx, tet in enumerate(delaunay.simplices):
            verts = self.positions[tet]
            circumradius = self._compute_circumradius(verts)
            if circumradius <= self.probe_radius:
                valid_tets.append(tet_idx)
        return np.array(valid_tets)
    
    def _compute_circumradius(self, vertices):
        """Calcula circumradius de tetraedro"""
        v0, v1, v2, v3 = vertices
        a = v1 - v0
        b = v2 - v0
        c = v3 - v0
        volume = abs(np.dot(a, np.cross(b, c))) / 6.0
        
        if volume < 1e-12:
            return np.inf
        
        A = np.array([2*(v1 - v0), 2*(v2 - v0), 2*(v3 - v0)])
        b_vec = np.array([
            np.dot(v1, v1) - np.dot(v0, v0),
            np.dot(v2, v2) - np.dot(v0, v0),
            np.dot(v3, v3) - np.dot(v0, v0)
        ])
        
        try:
            center = np.linalg.solve(A, b_vec)
            R = np.linalg.norm(center - v0)
            return R
        except np.linalg.LinAlgError:
            return np.inf
    
    def _extract_surface_facets(self, delaunay, valid_tets):
        """Extrae facetas de superficie"""
        valid_tet_set = set(valid_tets)
        facet_to_tets = {}
        
        for tet_idx, tet in enumerate(delaunay.simplices):
            is_valid = tet_idx in valid_tet_set
            for i in range(4):
                facet = tuple(sorted(np.delete(tet, i)))
                facet_key = frozenset(facet)
                if facet_key not in facet_to_tets:
                    facet_to_tets[facet_key] = []
                facet_to_tets[facet_key].append((tet_idx, is_valid))
        
        surface_facets = []
        for facet_key, tet_list in facet_to_tets.items():
            valid_count = sum(1 for _, is_valid in tet_list if is_valid)
            if valid_count == 1:
                surface_facets.append(list(facet_key))
        
        return surface_facets
    
    def _build_mesh(self, delaunay, surface_facets):
        """Construye malla de superficie"""
        if not surface_facets:
            self._surface_atom_indices = np.array([], dtype=int)
            return np.array([]), np.array([])
        
        surface_vertex_indices = sorted(set(np.array(surface_facets).flatten()))
        self._surface_atom_indices = np.array(surface_vertex_indices, dtype=int)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(surface_vertex_indices)}
        
        vertices = self.positions[surface_vertex_indices]
        faces = [[vertex_map[v] for v in facet] for facet in surface_facets]
        
        return vertices, np.array(faces)
    
    def _compute_surface_area(self):
        """Calcula área de superficie"""
        if self.surface_faces is None or len(self.surface_faces) == 0:
            return 0.0
        
        total_area = 0.0
        for face in self.surface_faces:
            if len(face) == 3:
                v0, v1, v2 = self.surface_vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                total_area += area
        
        return total_area
    
    def filter_surface_atoms_exclude_box_boundary(self, box_bounds=None, margin=0.01):
        """Filtra átomos de superficie excluyendo borde"""
        if len(self._surface_atom_indices) == 0:
            return np.array([], dtype=int)
        
        if box_bounds is None:
            box_bounds = (
                (self.positions[:, 0].min(), self.positions[:, 0].max()),
                (self.positions[:, 1].min(), self.positions[:, 1].max()),
                (self.positions[:, 2].min(), self.positions[:, 2].max())
            )
        
        interior_atoms = []
        for atom_idx in self._surface_atom_indices:
            pos = self.positions[atom_idx]
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


# ==========================================
# MÓDULO 2: CLUSTERING JERÁRQUICO
# ==========================================

class MeanShiftClusterer:
    """Clustering con Mean Shift o KMeans"""
    
    def __init__(self, data_tuple):
        self.header, self.data_df = data_tuple
        self.coords = self.data_df[['x', 'y', 'z']].values
        self.labels = None
    
    def aplicar_clustering(self, n_clusters=None, quantile=0.2):
        """Aplica clustering"""
        if n_clusters is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels = kmeans.fit_predict(self.coords)
            n_clusters_found = n_clusters
        else:
            bandwidth = estimate_bandwidth(
                self.coords, 
                quantile=quantile, 
                n_samples=min(500, len(self.coords))
            )
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            self.labels = ms.fit_predict(self.coords)
            n_clusters_found = len(np.unique(self.labels))
        
        self.data_df['Cluster'] = self.labels
        return n_clusters_found


class HierarchicalMeanShiftClusterer:
    """Clustering jerárquico"""
    
    def __init__(self):
        self.final_clusters = []
        self.cluster_counter = 0
    
    def calcular_metricas(self, coords, labels):
        """Calcula métricas de clustering"""
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and len(coords) > n_clusters:
            try:
                return {
                    'silhouette': silhouette_score(coords, labels),
                    'davies_bouldin': davies_bouldin_score(coords, labels),
                }
            except:
                return {}
        return {}
    
    def evaluar_subdivision(self, coords, labels, min_atoms, sil_thresh=0.3, db_thresh=1.5):
        """Evalúa si un cluster debe ser subdividido"""
        n_atoms = len(coords)
        n_clusters = len(np.unique(labels))
        
        if n_atoms < min_atoms * 2:
            return False, f"Pocos átomos ({n_atoms})", {}
        
        if n_clusters == 1:
            metricas = self.calcular_metricas(coords, labels)
            return n_atoms >= min_atoms * 3, "Cluster único", metricas
        
        metricas = self.calcular_metricas(coords, labels)
        razones = []
        necesita_subdivision = False
        
        if metricas.get('silhouette', -1) < sil_thresh:
            razones.append(f"Silhouette bajo ({metricas['silhouette']:.3f} < {sil_thresh})")
            necesita_subdivision = True
        
        if metricas.get('davies_bouldin', float('inf')) > db_thresh:
            razones.append(f"Davies-Bouldin alto ({metricas['davies_bouldin']:.3f} > {db_thresh})")
            necesita_subdivision = True
        
        razon = " | ".join(razones) if razones else "Métricas aceptables"
        
        return necesita_subdivision, razon, metricas
    
    def clustering_recursivo(self, data_tuple, nivel=0, min_atoms=50, 
                            max_iteraciones=5, n_clusters_target=None, quantile=0.2,
                            sil_thresh=0.3, db_thresh=1.5):
        """Clustering recursivo"""
        if nivel >= max_iteraciones:
            self.cluster_counter += 1
            header, df = data_tuple
            self.final_clusters.append({
                'data_tuple': (header, df),
                'nombre': f"cluster_{self.cluster_counter:03d}",
                'n_atoms': len(df),
                'nivel': nivel,
                'razon': 'Nivel máximo alcanzado'
            })
            return
        
        header, df = data_tuple
        n_atoms = len(df)
        
        if n_atoms < min_atoms * 2:
            self.cluster_counter += 1
            self.final_clusters.append({
                'data_tuple': (header, df),
                'nombre': f"cluster_{self.cluster_counter:03d}",
                'n_atoms': n_atoms,
                'nivel': nivel,
                'razon': f'Pocos átomos ({n_atoms})'
            })
            return
        
        clusterer = MeanShiftClusterer(data_tuple)
        n_clusters_test = n_clusters_target or (2 if n_atoms > 1000 else 3)
        
        if n_atoms < n_clusters_test * min_atoms:
            n_clusters_test = max(2, n_atoms // min_atoms)
        
        clusterer.aplicar_clustering(n_clusters=n_clusters_test, quantile=quantile)
        
        necesita_subdivision, razon, metricas = self.evaluar_subdivision(
            clusterer.coords, clusterer.labels, min_atoms,
            sil_thresh=sil_thresh, db_thresh=db_thresh
        )
        
        if not necesita_subdivision:
            self.cluster_counter += 1
            self.final_clusters.append({
                'data_tuple': (header, clusterer.data_df.drop(columns=['Cluster'])),
                'nombre': f"cluster_{self.cluster_counter:03d}",
                'n_atoms': n_atoms,
                'nivel': nivel,
                'razon': razon,
                'metricas': metricas
            })
            return
        
        unique_labels = np.unique(clusterer.labels)
        for label in unique_labels:
            subcluster_df = clusterer.data_df[clusterer.data_df['Cluster'] == label].copy()
            subcluster_df.drop(columns=['Cluster'], inplace=True)
            self.clustering_recursivo(
                (header, subcluster_df), nivel + 1, min_atoms, max_iteraciones,
                n_clusters_target, quantile, sil_thresh, db_thresh
            )


# ==========================================
# INTERFAZ STREAMLIT
# ==========================================

# PASO 1: UPLOAD
st.header("Paso 1: Cargar archivo LAMMPS")

uploaded_file = st.file_uploader("Selecciona archivo .dump", type=['dump', 'txt'])

if uploaded_file:
    # Parse una sola vez y guarda en session_state
    if st.session_state.pipeline_state['original_data'] is None:
        try:
            header, df = parse_lammps_dump(uploaded_file.getvalue())
            st.session_state.pipeline_state['original_data'] = (header, df)
            st.rerun()
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")
    
    header, df = st.session_state.pipeline_state['original_data']
    
    st.success(f"Archivo cargado: {len(df)} átomos")
    
    # PASO 2: ALPHA SHAPE
    st.header("Paso 2: Filtrado de Nanoporos (Alpha Shape)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        probe_radius = st.number_input("Radio de sonda (Å)", value=2.0, step=0.5)
    with col2:
        smoothing = st.number_input("Suavizado", value=1, step=1)
    with col3:
        boundary_margin = st.number_input("Margen de borde (Å)", value=0.1, step=0.05)
    
    if st.button("Ejecutar Alpha Shape"):
        with st.spinner("Procesando Alpha Shape..."):
            try:
                positions = df[['x', 'y', 'z']].values
                box_bounds = tuple(
                    (header['box_bounds'][i][0], header['box_bounds'][i][1])
                    for i in range(3)
                )
                
                constructor = AlphaShapeSurfaceConstructor(
                    positions=positions,
                    probe_radius=probe_radius,
                    smoothing_level=smoothing
                )
                constructor.perform()
                
                surface_atoms = constructor.filter_surface_atoms_exclude_box_boundary(
                    box_bounds=box_bounds,
                    margin=boundary_margin
                )
                
                filtered_df = df.iloc[surface_atoms].copy()
                
                st.session_state.pipeline_state['alpha_done'] = True
                st.session_state.pipeline_state['filtered_data'] = (header, filtered_df)
                st.session_state.pipeline_state['step'] = 2
                
                st.success(f"Alpha Shape completado: {len(surface_atoms)} átomos detectados")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
    
    # Mostrar resultados de Alpha Shape
    if st.session_state.pipeline_state['alpha_done']:
        _, filtered_df = st.session_state.pipeline_state['filtered_data']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Átomos detectados", len(filtered_df))
        col2.metric("Átomos originales", len(df))
        col3.metric("Porcentaje", f"{100*len(filtered_df)/len(df):.1f}%")
        
        # VISUALIZACIÓN 3D - NANOPOROS
        with st.expander("Visualización 3D - Nanoporos Detectados"):
            df_all_atoms = df.copy()
            df_all_atoms['Tipo'] = 'Bulk'
            
            df_filtered_copy = filtered_df.copy()
            df_filtered_copy['Tipo'] = 'Nanoporo'
            
            df_combined = pd.concat([df_all_atoms, df_filtered_copy], ignore_index=True)
            
            fig = px.scatter_3d(
                df_combined,
                x='x', y='y', z='z',
                color='Tipo',
                color_discrete_map={'Bulk': 'lightblue', 'Nanoporo': 'red'},
                title='Átomos de nanoporos (rojo) vs Bulk (azul)',
                labels={'x': 'X (Å)', 'y': 'Y (Å)', 'z': 'Z (Å)'},
            )
            
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        
        # PASO 3: CLUSTERING
        st.header("Paso 3: Clustering Jerárquico")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_atoms_cluster = st.number_input("Mín. átomos por cluster", value=10, step=5)
        with col2:
            max_levels = st.number_input("Máx. niveles", value=4, step=1)
        with col3:
            quantile_val = st.slider("Quantile", 0.05, 0.95, 0.2)
        
        # Parámetros avanzados de clustering
        with st.expander("Parámetros Avanzados de Subdivisión"):
            st.info("Estos umbrales deciden si un clúster debe ser subdividido recursivamente.")
            
            col1, col2 = st.columns(2)
            with col1:
                sil_threshold = st.slider(
                    "Umbral Silhouette",
                    min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                    help="Valor más bajo = más subdivisión. Mide qué tan bien separados están los clusters."
                )
            with col2:
                db_threshold = st.slider(
                    "Umbral Davies-Bouldin",
                    min_value=0.1, max_value=5.0, value=1.5, step=0.1,
                    help="Valor más bajo = mejor separación. Mide la densidad y separación entre clusters."
                )
            
            # Información visual sobre las métricas
            st.markdown("#### Interpretación de Métricas:")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Silhouette** (-1 a 1):
                - Cercano a 1: clusters bien separados
                - Cercano a 0: clusters solapados
                - Cercano a -1: puntos mal asignados
                """)
            with col2:
                st.markdown("""
                **Davies-Bouldin** (≥0):
                - Valores bajos: mejor separación
                - Típicamente 0.5-3.0
                - Penaliza densidad baja
                """)
        
        if st.button("Ejecutar Clustering"):
            with st.spinner("Procesando clustering jerárquico..."):
                try:
                    hierarchical = HierarchicalMeanShiftClusterer()
                    hierarchical.clustering_recursivo(
                        st.session_state.pipeline_state['filtered_data'],
                        nivel=0,
                        min_atoms=min_atoms_cluster,
                        max_iteraciones=max_levels,
                        quantile=quantile_val,
                        sil_thresh=sil_threshold,
                        db_thresh=db_threshold
                    )
                    
                    st.session_state.pipeline_state['clustering_done'] = True
                    st.session_state.pipeline_state['clustering_result'] = hierarchical
                    st.session_state.pipeline_state['step'] = 3
                    
                    st.success(f"Clustering completado: {len(hierarchical.final_clusters)} clusters")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error en clustering: {e}")
                    st.code(traceback.format_exc())
        
        # Resultados de clustering
        if st.session_state.pipeline_state['clustering_done']:
            clustering = st.session_state.pipeline_state['clustering_result']
            
            st.subheader("Resultados de Clustering")
            
            cluster_data = []
            for i, cluster in enumerate(clustering.final_clusters, 1):
                cluster_data.append({
                    'Cluster': i,
                    'Átomos': cluster['n_atoms'],
                    'Nivel': cluster['nivel']
                })
            
            df_clusters = pd.DataFrame(cluster_data)
            st.dataframe(df_clusters, use_container_width=True)
            
            col1, col2 = st.columns(2)
            col1.metric("Total clusters", len(clustering.final_clusters))
            col2.metric("Total átomos", sum(c['n_atoms'] for c in clustering.final_clusters))
            
            # Mostrar tabla con razones de parada
            st.subheader("Detalles de Clusters Finales")
            cluster_details = []
            for i, cluster in enumerate(clustering.final_clusters, 1):
                cluster_details.append({
                    'ID': i,
                    'Átomos': cluster['n_atoms'],
                    'Nivel': cluster['nivel'],
                    'Razón': cluster.get('razon', 'N/A')
                })
            
            df_details = pd.DataFrame(cluster_details)
            st.dataframe(df_details, use_container_width=True)
            
            # Mostrar estadísticas de distribución
            st.subheader("Estadísticas de Distribución")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min átomos", int(df_details['Átomos'].min()))
            col2.metric("Max átomos", int(df_details['Átomos'].max()))
            col3.metric("Promedio", int(df_details['Átomos'].mean()))
            col4.metric("Desv. Estándar", int(df_details['Átomos'].std()))
            
            # VISUALIZACIÓN 3D - CLUSTERS
            with st.expander("Visualización 3D de Clusters"):
                df_viz_list = []
                for i, cluster in enumerate(clustering.final_clusters, 1):
                    header_c, df_c = cluster['data_tuple']
                    df_c_copy = df_c.copy()
                    df_c_copy['Cluster'] = f"Cluster {i}"
                    df_viz_list.append(df_c_copy)
                
                df_viz = pd.concat(df_viz_list, ignore_index=True)
                
                fig = px.scatter_3d(
                    df_viz,
                    x='x', y='y', z='z',
                    color='Cluster',
                    title='Distribución espacial de clusters',
                    labels={'x': 'X (Å)', 'y': 'Y (Å)', 'z': 'Z (Å)'},
                    opacity=0.8
                )
                
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            # EXPORTACIÓN
            st.header("Exportación de Resultados")
            
            col1, col2 = st.columns(2)
            with col1:
                export_dir = st.text_input("Directorio de salida", value="results_pipeline")
            with col2:
                export_summary = st.checkbox("Incluir resumen JSON", value=True)
            
            if st.button("Exportar clusters"):
                try:
                    output_path = Path(export_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    for i, cluster in enumerate(clustering.final_clusters, 1):
                        output_file = output_path / f"cluster_{i:03d}.dump"
                        header_c, df_c = cluster['data_tuple']
                        write_lammps_dump(str(output_file), header_c, df_c)
                    
                    if export_summary:
                        summary = {
                            'total_clusters': len(clustering.final_clusters),
                            'total_atoms': sum(c['n_atoms'] for c in clustering.final_clusters),
                            'clusters': [
                                {
                                    'id': i,
                                    'atoms': c['n_atoms'],
                                    'level': c['nivel'],
                                    'file': f"cluster_{i:03d}.dump"
                                }
                                for i, c in enumerate(clustering.final_clusters, 1)
                            ]
                        }
                        
                        with open(output_path / "summary.json", 'w') as f:
                            json.dump(summary, f, indent=2)
                    
                    st.success(f"Exportados {len(clustering.final_clusters)} clusters")
                    
                    zip_path = Path(export_dir) / "clusters"
                    shutil.make_archive(str(zip_path), 'zip', export_dir)
                    
                    with open(f"{zip_path}.zip", "rb") as f:
                        st.download_button(
                            label="Descargar ZIP",
                            data=f,
                            file_name="clusters.zip",
                            mime="application/zip"
                        )
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

else:
    st.info("Carga un archivo .dump para comenzar el pipeline")