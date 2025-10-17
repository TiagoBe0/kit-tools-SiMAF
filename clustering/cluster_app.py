import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback
import shutil
from pathlib import Path
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any
import py3Dmol
import tempfile
from stmol import showmol
import time
import numpy as np
# ======================================
#  CONFIGURACIÓN DE PÁGINA
# ======================================
st.set_page_config(page_title="Clustering Jerárquico", page_icon="🧩", layout="wide")
st.title("🧩 Clustering Jerárquico para Archivos LAMMPS")
st.markdown("---")

# ======================================
#  MÓDULO DE LÓGICA DE CLUSTERING MODIFICADO
# ======================================

def _parse_lammps_dump(dump_file: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Lee un archivo dump de LAMMPS de forma robusta, extrayendo el encabezado y 
    filtrando únicamente los datos de los átomos.
    """
    header = {'box_bounds': []}
    atom_lines = []
    
    with open(dump_file, 'r') as f:
        line = f.readline()
        if "TIMESTEP" not in line:
            raise ValueError("Formato de archivo dump inválido: no se encontró 'ITEM: TIMESTEP'")
        
        header['timestep'] = int(f.readline())
        
        f.readline()
        header['n_atoms_header'] = int(f.readline())
        
        f.readline()
        
        for _ in range(3):
            line = f.readline().strip()
            line = line.replace('0.00.0', '0.0 0.0').replace('105.60.0', '105.6 0.0')
            parts = [float(x) for x in line.split()]
            
            if parts:
                lo = min(parts)
                hi = max(parts)
                header['box_bounds'].append([lo, hi])
            else:
                 header['box_bounds'].append([0.0, 0.0])

        atom_columns_line = f.readline()
        columns = atom_columns_line.strip().split()[2:]

        for line in f:
            line = line.strip()
            if line and not line.startswith("ITEM:") and not line.startswith("["):
                parts = line.split()
                if len(parts) == len(columns):
                    atom_lines.append(line + '\n')

    if not atom_lines:
        raise ValueError("No se encontraron líneas de datos de átomos válidas en el archivo.")
        
    data_io = StringIO("".join(atom_lines))
    df = pd.read_csv(data_io, delim_whitespace=True, names=columns)

    if len(df) != header['n_atoms_header']:
        st.warning(f"⚠️ Advertencia: El encabezado indica {header['n_atoms_header']} átomos, pero se leyeron {len(df)}.")
        
    return header, df

def _write_lammps_dump(output_path: str, header: Dict[str, Any], df: pd.DataFrame):
    """
    Escribe un DataFrame de Pandas en un archivo dump con formato LAMMPS.
    """
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

class MeanShiftClusterer:
    def __init__(self, dump_file=None, data_tuple=None):
        self.dump_file = dump_file
        self.header = None
        self.data_df = None
        self.coords = None
        self.labels = None
        
        if data_tuple:
            self.header, self.data_df = data_tuple

    def leer_dump(self):
        if self.data_df is not None:
            pass
        else:
            self.header, self.data_df = _parse_lammps_dump(self.dump_file)
        
        self.coords = self.data_df[['x', 'y', 'z']].values
        return len(self.coords)
    
    def aplicar_clustering(self, n_clusters=None, bandwidth=None, quantile=0.2):
        if n_clusters is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels = kmeans.fit_predict(self.coords)
            n_clusters_found = n_clusters
        else:
            if bandwidth is None:
                bandwidth = estimate_bandwidth(
                    self.coords, 
                    quantile=quantile, 
                    n_samples=min(500, len(self.coords))
                )
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            self.labels = ms.fit_predict(self.coords)
            n_clusters_found = len(np.unique(self.labels))
        
        # AÑADIDO: Guardar información del cluster en el DataFrame
        self.data_df['Cluster'] = self.labels
        self.data_df['Cluster_Level'] = 0  # Se actualizará en el proceso recursivo
        
        return n_clusters_found

# ======================================
#  VISUALIZACIÓN 3D
# ======================================
import matplotlib.pyplot as plt
import colorsys
import os
import tempfile
import numpy as np
import colorsys
import py3Dmol

def generate_distinct_colors(n):
    """
    Genera 'n' colores distintos en formato hexadecimal (#RRGGBB)
    usando el espacio HSV y colormap de matplotlib.
    """
    if n <= 12:
        # Paleta básica si son pocos clusters
        base_colors = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "brown", "gray", "black", "cyan", "magenta"
        ]
        return base_colors[:n]
    else:
        # Paleta dinámica HSV → RGB
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = [colorsys.hsv_to_rgb(h, 0.8, 0.9) for h in hues]
        return ["#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

def create_3d_visualization(df, title="Visualización 3D de Clusters"):
    """
    Crea una visualización 3D interactiva de los átomos coloreados por cluster,
    con colores únicos para cada cluster.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        f.write(f"{len(df)}\n")
        f.write(f"{title}\n")
        for _, atom in df.iterrows():
            f.write(f"C {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")
        temp_file = f.name

    viewer = py3Dmol.view(width=800, height=600)
    viewer.setBackgroundColor('white')

    with open(temp_file, 'r') as f:
        xyz_data = f.read()

    clusters = sorted(df['Cluster'].unique())
    colors = generate_distinct_colors(len(clusters))

    for i, cluster in enumerate(clusters):
        color = colors[i]
        subset = df[df['Cluster'] == cluster]
        xyz_str = f"{len(subset)}\nCluster {cluster}\n"
        for _, atom in subset.iterrows():
            xyz_str += f"C {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n"
        viewer.addModel(xyz_str, 'xyz')
        viewer.setStyle({'model': i+1}, {'sphere': {'color': color, 'radius': 0.45}})

    viewer.zoomTo()
    os.unlink(temp_file)
    return viewer

import asyncio
from stmol import showmol

import streamlit.components.v1 as components

def render_3d_viewer(viewer):
    """Renderiza el visualizador 3D en Streamlit (sin usar stmol)"""
    html = viewer._make_html()  # genera el HTML embebido de py3Dmol
    components.html(html, height=600, width=800)
# ======================================
#  CLUSTERING JERÁRQUICO MODIFICADO
# ======================================

class HierarchicalMeanShiftClusterer:
    def __init__(self):
        self.final_clusters = []
        self.cluster_counter = 0
        self.visualization_data = []  # Almacenar datos para visualización

    def calcular_metricas_clustering(self, coords, labels):
        metricas = {}
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and len(coords) > n_clusters:
            try:
                metricas['silhouette'] = silhouette_score(coords, labels)
                metricas['davies_bouldin'] = davies_bouldin_score(coords, labels)
                metricas['calinski_harabasz'] = calinski_harabasz_score(coords, labels)
                dispersiones = []
                for label in np.unique(labels):
                    cluster_points = coords[labels == label]
                    if len(cluster_points) > 1:
                        centroid = cluster_points.mean(axis=0)
                        dispersiones.append(np.mean(np.linalg.norm(cluster_points - centroid, axis=1)))
                metricas['dispersion_promedio'] = np.mean(dispersiones) if dispersiones else 0
            except Exception as e:
                metricas = self._metricas_default()
        else:
            metricas = self._metricas_default()
        return metricas

    def _metricas_default(self):
        return {'silhouette': -1, 'davies_bouldin': float('inf'), 'calinski_harabasz': 0, 'dispersion_promedio': float('inf')}

    def evaluar_necesidad_subdivision(self, coords, labels, min_atoms, 
                                     silhouette_threshold=0.3,
                                     davies_bouldin_threshold=1.5,
                                     dispersion_threshold=None):
        n_atoms = len(coords)
        n_clusters = len(np.unique(labels))
        
        if n_atoms < min_atoms * 2:
            return False, f"Muy pocos átomos ({n_atoms} < {min_atoms * 2})", {}

        if n_clusters == 1:
            dispersion = np.std(coords, axis=0).mean()
            if dispersion_threshold and dispersion > dispersion_threshold:
                return True, f"Alta dispersión en cluster único ({dispersion:.2f})", {'dispersion': dispersion}
            if n_atoms >= min_atoms * 3:
                return True, f"Cluster único con {n_atoms} átomos", {'dispersion': dispersion}
            return False, "Cluster único compacto", {'dispersion': dispersion}

        metricas = self.calcular_metricas_clustering(coords, labels)
        razones = []
        necesita_subdivision = False
        
        if metricas['silhouette'] < silhouette_threshold:
            razones.append(f"Silhouette bajo ({metricas['silhouette']:.3f} < {silhouette_threshold})")
            necesita_subdivision = True
        if metricas['davies_bouldin'] > davies_bouldin_threshold:
            razones.append(f"Davies-Bouldin alto ({metricas['davies_bouldin']:.3f} > {davies_bouldin_threshold})")
            necesita_subdivision = True
        if dispersion_threshold and metricas['dispersion_promedio'] > dispersion_threshold:
            razones.append(f"Alta dispersión ({metricas['dispersion_promedio']:.3f} > {dispersion_threshold})")
            necesita_subdivision = True
            
        return necesita_subdivision, " | ".join(razones) if razones else "Métricas aceptables", metricas

    def clustering_recursivo_memoria(self, data_input, nivel=0, 
                                    min_atoms=50, max_iterations=5, 
                                    n_clusters_target=None,
                                    silhouette_threshold=0.3,
                                    davies_bouldin_threshold=1.5,
                                    dispersion_threshold=None,
                                    quantile=0.2,
                                    parent_cluster_id=0):
        
        if nivel >= max_iterations:
            self.cluster_counter += 1
            header, df = data_input if isinstance(data_input, tuple) else _parse_lammps_dump(data_input)
            
            # AÑADIDO: Actualizar información de cluster para visualización
            df['Cluster'] = self.cluster_counter
            df['Cluster_Level'] = nivel
            
            self.final_clusters.append({
                'data_tuple': (header, df),
                'nombre': f"cluster_final_{self.cluster_counter:03d}",
                'n_atoms': len(df),
                'nivel': nivel,
                'razon_final': 'Nivel máximo alcanzado'
            })
            
            # AÑADIDO: Guardar para visualización
            self.visualization_data.append(df.copy())
            
            return {'subdivided': False, 'razon': 'Nivel máximo alcanzado'}

        clusterer = MeanShiftClusterer(data_tuple=data_input if isinstance(data_input, tuple) else None,
                                     dump_file=data_input if isinstance(data_input, str) else None)
        
        n_atoms = clusterer.leer_dump()
        
        if n_atoms < min_atoms * 2:
            self.cluster_counter += 1
            
            # AÑADIDO: Actualizar información de cluster
            clusterer.data_df['Cluster'] = self.cluster_counter
            clusterer.data_df['Cluster_Level'] = nivel
            
            self.final_clusters.append({
                'data_tuple': (clusterer.header, clusterer.data_df),
                'nombre': f"cluster_final_{self.cluster_counter:03d}",
                'n_atoms': n_atoms,
                'nivel': nivel,
                'razon_final': f'Pocos átomos ({n_atoms})'
            })
            
            # AÑADIDO: Guardar para visualización
            self.visualization_data.append(clusterer.data_df.copy())
            
            return {'subdivided': False, 'razon': f'Pocos átomos ({n_atoms})'}

        n_clusters_test = n_clusters_target or (2 if n_atoms > 1000 else 3)
        if n_atoms < n_clusters_test * min_atoms:
            n_clusters_test = max(2, n_atoms // min_atoms)

        clusterer.aplicar_clustering(n_clusters=n_clusters_test, quantile=quantile)
        
        # AÑADIDO: Actualizar nivel de cluster
        clusterer.data_df['Cluster_Level'] = nivel
        
        necesita_subdivision, razon, metricas = self.evaluar_necesidad_subdivision(
            clusterer.coords, clusterer.labels, min_atoms,
            silhouette_threshold, davies_bouldin_threshold, dispersion_threshold
        )
        
        if not necesita_subdivision:
            self.cluster_counter += 1
            
            # AÑADIDO: Reasignar cluster único para este nivel
            clusterer.data_df['Cluster'] = self.cluster_counter
            
            self.final_clusters.append({
                'data_tuple': (clusterer.header, clusterer.data_df.drop(columns=['Cluster'])),
                'nombre': f"cluster_final_{self.cluster_counter:03d}",
                'n_atoms': n_atoms,
                'nivel': nivel,
                'razon_final': razon,
                'metricas': metricas
            })
            
            # AÑADIDO: Guardar para visualización
            self.visualization_data.append(clusterer.data_df.copy())
            
            return {'subdivided': False, 'razon': razon}

        # AÑADIDO: Guardar estado actual para visualización
        self.visualization_data.append(clusterer.data_df.copy())
        
        unique_labels = np.unique(clusterer.labels)
        
        for i, label in enumerate(unique_labels):
            subcluster_df = clusterer.data_df[clusterer.data_df['Cluster'] == label].copy()
            subcluster_df.drop(columns=['Cluster'], inplace=True)
            
            self.clustering_recursivo_memoria(
                (clusterer.header, subcluster_df), nivel + 1,
                min_atoms, max_iterations, n_clusters_target,
                silhouette_threshold, davies_bouldin_threshold,
                dispersion_threshold, quantile,
                parent_cluster_id=self.cluster_counter
            )

    def exportar_clusters_finales(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.final_clusters.sort(key=lambda x: x['n_atoms'], reverse=True)
        
        for i, cluster_info in enumerate(self.final_clusters, 1):
            cluster_name = f"cluster_{i:03d}"
            output_file = output_path / f"{cluster_name}.dump"
            header, df = cluster_info['data_tuple']
            
            _write_lammps_dump(str(output_file), header, df)
            
            cluster_info['nombre_final'] = cluster_name
            cluster_info['archivo_final'] = str(output_file)

    def get_visualization_data(self):
        """Obtener todos los datos para visualización"""
        if self.visualization_data:
            return pd.concat(self.visualization_data, ignore_index=True)
        return pd.DataFrame()

def clustering_jerarquico_final(dump_file, output_dir="clusters_final",
                               min_atoms=50, max_iterations=5,
                               n_clusters_per_level=None,
                               silhouette_threshold=0.3,
                               davies_bouldin_threshold=1.5,
                               dispersion_threshold=None,
                               quantile=0.2,
                               limpiar_intermedios=True,
                               enable_visualization=True):
    
    output_path = Path(output_dir)
    if limpiar_intermedios and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hierarchical_clusterer = HierarchicalMeanShiftClusterer()
    
    hierarchical_clusterer.clustering_recursivo_memoria(
        dump_file, nivel=0,
        min_atoms=min_atoms, max_iterations=max_iterations,
        n_clusters_target=n_clusters_per_level,
        silhouette_threshold=silhouette_threshold,
        davies_bouldin_threshold=davies_bouldin_threshold,
        dispersion_threshold=dispersion_threshold,
        quantile=quantile
    )
    
    hierarchical_clusterer.exportar_clusters_finales(output_dir)
    
    # CORRECCIÓN: Crear resumen sin el DataFrame para JSON
    resumen_para_json = {
        'archivo_original': dump_file,
        'n_clusters_finales': len(hierarchical_clusterer.final_clusters),
        'clusters_finales': []
    }
    
    # Resumen completo para retornar (incluye datos de visualización)
    resumen_completo = {
        'archivo_original': dump_file,
        'n_clusters_finales': len(hierarchical_clusterer.final_clusters),
        'clusters_finales': [],
        'visualization_data': None
    }
    
    # Llenar ambos resúmenes con la información de clusters
    for cluster in hierarchical_clusterer.final_clusters:
        cluster_info = {
            'nombre': cluster.get('nombre_final', ''),
            'archivo': cluster.get('archivo_final', ''),
            'n_atoms': cluster.get('n_atoms', 0),
            'nivel': cluster.get('nivel', 0),
            'razon_final': cluster.get('razon_final', ''),
            'metricas': cluster.get('metricas', {})
        }
        
        resumen_para_json['clusters_finales'].append(cluster_info)
        resumen_completo['clusters_finales'].append(cluster_info)
    
    # CORRECCIÓN: Solo agregar datos de visualización si está habilitado
    if enable_visualization:
        visualization_df = hierarchical_clusterer.get_visualization_data()
        resumen_completo['visualization_data'] = visualization_df
        
        # Guardar datos de visualización en un archivo CSV separado
        if not visualization_df.empty:
            csv_path = output_path / "visualization_data.csv"
            visualization_df.to_csv(csv_path, index=False)
    
    # CORRECCIÓN: Guardar solo el resumen sin DataFrame en JSON
    json_path = output_path / "clustering_summary.json"
    with open(json_path, 'w') as f:
        json.dump(resumen_para_json, f, indent=2, default=str)
    
    return resumen_completo

# ======================================
#  INTERFAZ DE STREAMLIT MEJORADA
# ======================================

# SIDEBAR DE PARÁMETROS
st.sidebar.header("1. Cargar Archivo")
uploaded_file = st.sidebar.file_uploader(
    "Subir archivo .dump",
    type=['dump', 'txt', 'track_clustering'],
    help="Sube tu archivo .dump de LAMMPS"
)

st.sidebar.header("2. Configuración del Clustering")

min_atomos = st.sidebar.number_input("Mínimo de átomos por clúster final", min_value=10, value=30, step=5)
max_niveles = st.sidebar.number_input("Máximo de niveles de recursión", min_value=1, max_value=10, value=4, step=1)
output_dir = st.sidebar.text_input("Directorio de salida", value="clusters_finales_streamlit")

# AÑADIDO: Opción para visualización en tiempo real
enable_live_visualization = st.sidebar.checkbox("Visualización 3D en vivo", value=True, 
                                               help="Mostrar visualización 3D durante el proceso de clustering")

with st.sidebar.expander("Parámetros avanzados de subdivisión"):
    st.info("Estos umbrales deciden si un clúster debe ser subdividido.")
    umbral_silhouette = st.slider("Umbral Silhouette", 0.0, 1.0, 0.3)
    umbral_davies_bouldin = st.slider("Umbral Davies-Bouldin", 0.1, 5.0, 1.5)
    umbral_dispersion = st.number_input("Umbral de dispersión", min_value=0.0, value=5.0, step=0.5)
    quantile = st.slider("Quantile (para estimación inicial)", 0.05, 0.95, 0.2)

st.sidebar.header("3. Ejecutar")
run_clustering = st.sidebar.button("🚀 Iniciar Proceso de Clustering")

# INTERFAZ PRINCIPAL
if not uploaded_file:
    st.info("👋 ¡Bienvenido! Por favor, sube un archivo `.dump` usando la barra lateral para comenzar.")
    st.markdown("""
    **Nueva característica: Visualización 3D en vivo**
    - Ve la evolución del clustering en tiempo real
    - Átomos coloreados por cluster
    - Interactúa con la visualización 3D
    """)

if uploaded_file is not None:
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    temp_file_path = output_path / f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Archivo cargado: **{uploaded_file.name}**")
    
    if run_clustering:
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.info("Iniciando el proceso de clustering jerárquico...")

            params_data = {
                "Parámetro": ["Mín. átomos", "Máx. niveles", "Umbral Silhouette", "Umbral Davies-Bouldin", "Umbral Dispersión"],
                "Valor": [min_atomos, max_niveles, umbral_silhouette, umbral_davies_bouldin, umbral_dispersion]
            }
            st.table(pd.DataFrame(params_data))

            resumen = None
            with st.spinner("🧠 Analizando y subdividiendo clusters..."):
                resumen = clustering_jerarquico_final(
                    dump_file=str(temp_file_path),
                    output_dir=output_dir,
                    min_atoms=min_atomos,
                    max_iterations=max_niveles,
                    silhouette_threshold=umbral_silhouette,
                    davies_bouldin_threshold=umbral_davies_bouldin,
                    dispersion_threshold=umbral_dispersion,
                    quantile=quantile,
                    limpiar_intermedios=False,
                    enable_visualization=enable_live_visualization
                )

            st.success("✅ ¡Clustering completado exitosamente!")

            # AÑADIDO: Visualización final
            if enable_live_visualization and resumen.get('visualization_data') is not None and not resumen['visualization_data'].empty:
                st.header("🎯 Visualización 3D Final")
                final_df = resumen['visualization_data']
                
                # Mostrar estadísticas de clusters
                st.subheader("Resumen de Clusters")
                cluster_stats = final_df['Cluster'].value_counts().reset_index()
                cluster_stats.columns = ['Cluster', 'Número de Átomos']
                st.dataframe(cluster_stats)
                import streamlit.components.v1 as components

                # Visualización 3D
                st.subheader("Visualización 3D Interactiva")
                viewer = create_3d_visualization(final_df, "Resultado Final del Clustering")
                html = viewer._make_html()
                components.html(html, height=600, width=800)


            # Resto del código de resultados...
            if resumen:
                st.header("Resumen de Resultados")
                n_finales = resumen['n_clusters_finales']
                atomos_totales = sum(c['n_atoms'] for c in resumen['clusters_finales'])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Clusters Finales", f"{n_finales}")
                col2.metric("Átomos Clusterizados", f"{atomos_totales}")
                
                df_clusters = pd.DataFrame(resumen['clusters_finales'])
                df_clusters_display = df_clusters[['nombre', 'n_atoms', 'nivel', 'razon_final']].rename(
                    columns={'nombre': 'Clúster', 'n_atoms': 'Átomos', 'nivel': 'Nivel', 'razon_final': 'Razón de Parada'}
                )
                st.dataframe(df_clusters_display, use_container_width=True)

                st.subheader("Distribución del Tamaño de los Clusters")
                chart_data = df_clusters_display.set_index('Clúster')['Átomos']
                st.bar_chart(chart_data)

                st.header("Descargar Resultados")
                zip_path = output_path / "clusters_exportados.zip"
                shutil.make_archive(zip_path.with_suffix(""), "zip", output_dir)
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Descargar todos los clusters (.zip)",
                        data=f,
                        file_name="clusters_exportados.zip",
                        mime="application/zip",
                    )

        except Exception as e:
            st.error(f"❌ Ocurrió un error durante el clustering:")
            st.code(f"{e}\n\n{traceback.format_exc()}")

        finally:
            if temp_file_path.exists():
                os.remove(temp_file_path)