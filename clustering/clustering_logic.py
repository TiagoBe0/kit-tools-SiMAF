import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import os
import json
import shutil
from typing import Dict, List, Tuple, Optional, Any
from io import StringIO

# ==============================================================================
# FUNCIONES AUXILIARES PARA MANEJAR ARCHIVOS DUMP DE LAMMPS (SIN OVITO)
# ==============================================================================

def _parse_lammps_dump(dump_file: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Lee un archivo dump de LAMMPS de forma robusta, extrayendo el encabezado y 
    filtrando únicamente los datos de los átomos.
    
    Args:
        dump_file (str): Ruta al archivo .dump de entrada.
        
    Returns:
        Tuple[Dict[str, Any], pd.DataFrame]: 
            - Un diccionario con la información del encabezado.
            - Un DataFrame de Pandas con los datos de los átomos.
    """
    header = {'box_bounds': []}
    atom_lines = []
    
    with open(dump_file, 'r') as f:
        # 1. Leer y procesar el encabezado
        line = f.readline() # ITEM: TIMESTEP
        if "TIMESTEP" not in line:
            raise ValueError("Formato de archivo dump inválido: no se encontró 'ITEM: TIMESTEP'")
        
        header['timestep'] = int(f.readline())
        
        f.readline() # ITEM: NUMBER OF ATOMS
        header['n_atoms_header'] = int(f.readline())
        
        f.readline() # ITEM: BOX BOUNDS ...
        
        # --- INICIO DE LA CORRECCIÓN ---
        for _ in range(3):
            line = f.readline().strip()
            
            # Primero, corregimos los problemas de formato conocidos del archivo original
            # Esto asegura que los números pegados se separen con un espacio
            line = line.replace('0.00.0', '0.0 0.0').replace('105.60.0', '105.6 0.0')
            
            # Convertimos todas las partes numéricas de la línea a float
            parts = [float(x) for x in line.split()]
            
            # El límite inferior (lo) es el mínimo y el superior (hi) es el máximo
            # Esto es robusto y funciona para cualquier orden de números.
            if parts:
                lo = min(parts)
                hi = max(parts)
                header['box_bounds'].append([lo, hi])
            else:
                 # Si la línea está vacía o es inválida, usamos valores por defecto
                 header['box_bounds'].append([0.0, 0.0])
        # --- FIN DE LA CORRECCIÓN ---

        atom_columns_line = f.readline() # ITEM: ATOMS ...
        columns = atom_columns_line.strip().split()[2:]

        # 2. Leer el resto del archivo y filtrar solo las líneas de átomos
        for line in f:
            line = line.strip()
            if line and not line.startswith("ITEM:") and not line.startswith("["):
                parts = line.split()
                if len(parts) == len(columns):
                    atom_lines.append(line + '\n')

    # 3. Crear el DataFrame
    if not atom_lines:
        raise ValueError("No se encontraron líneas de datos de átomos válidas en el archivo.")
        
    data_io = StringIO("".join(atom_lines))
    df = pd.read_csv(data_io, delim_whitespace=True, names=columns)

    if len(df) != header['n_atoms_header']:
        print(f"⚠️ Advertencia: El encabezado indica {header['n_atoms_header']} átomos, pero se leyeron {len(df)}.")
        
    return header, df


def _write_lammps_dump(output_path: str, header: Dict[str, Any], df: pd.DataFrame):
    """
    Escribe un DataFrame de Pandas en un archivo dump con formato LAMMPS.
    
    Args:
        output_path (str): Ruta del archivo de salida.
        header (Dict[str, Any]): Diccionario con la información del encabezado.
        df (pd.DataFrame): DataFrame con los datos de los átomos a escribir.
    """
    with open(output_path, 'w') as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{header['timestep']}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(df)}\n") # Actualizar el número de átomos
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for bounds in header['box_bounds']:
            f.write(f"{bounds[0]} {bounds[1]}\n")
        
        f.write(f"ITEM: ATOMS {' '.join(df.columns)}\n")
        
        # Escribir los datos del DataFrame
        f.write(df.to_string(header=False, index=False, float_format="%.8f"))
        f.write("\n")

# ==============================================================================
# CLASES DE CLUSTERING ADAPTADAS A PANDAS (sin cambios desde aquí)
# ==============================================================================

class MeanShiftClusterer:
    """
    Clase base para clusterizar átomos usando Mean Shift o KMeans 
    y exportar clusters individuales. Versión sin OVITO.
    """
    
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
            print(f"📂 Usando datos en memoria (DataFrame)")
        else:
            print(f"📂 Leyendo archivo: {self.dump_file}")
            self.header, self.data_df = _parse_lammps_dump(self.dump_file)
        
        self.coords = self.data_df[['x', 'y', 'z']].values
        
        print(f"✓ Procesando {len(self.coords)} átomos")
        return len(self.coords)
    
    def aplicar_clustering(self, n_clusters=None, bandwidth=None, quantile=0.2):
        if n_clusters is not None:
            print(f"🔄 Aplicando KMeans clustering (n_clusters={n_clusters})...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.labels = kmeans.fit_predict(self.coords)
            n_clusters_found = n_clusters
        else:
            print("🔄 Aplicando Mean Shift clustering...")
            if bandwidth is None:
                bandwidth = estimate_bandwidth(
                    self.coords, 
                    quantile=quantile, 
                    n_samples=min(500, len(self.coords))
                )
                print(f"   Bandwidth estimado: {bandwidth:.3f}")
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            self.labels = ms.fit_predict(self.coords)
            n_clusters_found = len(np.unique(self.labels))
        
        print(f"✓ Encontrados {n_clusters_found} clusters")
        self.data_df['Cluster'] = self.labels
        
        for i in range(n_clusters_found):
            count = np.sum(self.labels == i)
            print(f"   Cluster {i}: {count} átomos")
        
        return n_clusters_found

    def exportar_clusters(self, output_dir="clusters_output"):
        if 'Cluster' not in self.data_df.columns:
            print("⌛ Error: Primero debes ejecutar aplicar_clustering()")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        unique_clusters = sorted(self.data_df['Cluster'].unique())
        base_name = Path(self.dump_file).stem if self.dump_file else "cluster"
        
        print(f"\n📁 Exportando clusters a {output_dir}/")
        
        for i, cluster_id in enumerate(unique_clusters):
            output_file = output_path / f"{base_name}_cluster_{i}.dump"
            cluster_df = self.data_df[self.data_df['Cluster'] == cluster_id].copy()
            cluster_df.drop(columns=['Cluster'], inplace=True)
            _write_lammps_dump(str(output_file), self.header, cluster_df)
            print(f"   ✓ {output_file.name} ({len(cluster_df)} átomos)")


class HierarchicalMeanShiftClusterer:
    def __init__(self):
        self.final_clusters = []
        self.cluster_counter = 0

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
                print(f"   ⚠️ Error calculando métricas: {e}")
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
                                    quantile=0.2):
        if nivel >= max_iterations:
            print(f"{'  ' * nivel}⚠️ Alcanzado nivel máximo de iteraciones ({max_iterations})")
            self.cluster_counter += 1
            header, df = data_input if isinstance(data_input, tuple) else _parse_lammps_dump(data_input)
            self.final_clusters.append({
                'data_tuple': (header, df),
                'nombre': f"cluster_final_{self.cluster_counter:03d}",
                'n_atoms': len(df),
                'nivel': nivel,
                'razon_final': 'Nivel máximo alcanzado'
            })
            return {'subdivided': False, 'razon': 'Nivel máximo alcanzado'}

        clusterer = MeanShiftClusterer(data_tuple=data_input if isinstance(data_input, tuple) else None,
                                     dump_file=data_input if isinstance(data_input, str) else None)
        
        if isinstance(data_input, str):
             print(f"\n{'  ' * nivel}📍 Nivel {nivel}: Procesando archivo {Path(data_input).name}")
        else:
             print(f"\n{'  ' * nivel}📍 Nivel {nivel}: Procesando datos en memoria")

        n_atoms = clusterer.leer_dump()
        
        if n_atoms < min_atoms * 2:
            print(f"{'  ' * nivel}✋ Cluster final: muy pocos átomos ({n_atoms})")
            self.cluster_counter += 1
            self.final_clusters.append({
                'data_tuple': (clusterer.header, clusterer.data_df),
                'nombre': f"cluster_final_{self.cluster_counter:03d}",
                'n_atoms': n_atoms,
                'nivel': nivel,
                'razon_final': f'Pocos átomos ({n_atoms})'
            })
            return {'subdivided': False, 'razon': f'Pocos átomos ({n_atoms})'}

        print(f"{'  ' * nivel}🔍 Evaluando necesidad de subdivisión...")
        n_clusters_test = n_clusters_target or (2 if n_atoms > 1000 else 3)
        if n_atoms < n_clusters_test * min_atoms:
            n_clusters_test = max(2, n_atoms // min_atoms)

        clusterer.aplicar_clustering(n_clusters=n_clusters_test, quantile=quantile)
        
        necesita_subdivision, razon, metricas = self.evaluar_necesidad_subdivision(
            clusterer.coords, clusterer.labels, min_atoms,
            silhouette_threshold, davies_bouldin_threshold, dispersion_threshold
        )
        
        print(f"{'  ' * nivel}📊 Métricas: {razon}")
        for key, value in metricas.items():
            if value != float('inf') and value != -1:
                print(f"{'  ' * nivel}   - {key}: {value:.3f}")
        
        if not necesita_subdivision:
            print(f"{'  ' * nivel}✅ Cluster final: {razon}")
            self.cluster_counter += 1
            self.final_clusters.append({
                'data_tuple': (clusterer.header, clusterer.data_df.drop(columns=['Cluster'])),
                'nombre': f"cluster_final_{self.cluster_counter:03d}",
                'n_atoms': n_atoms,
                'nivel': nivel,
                'razon_final': razon,
                'metricas': metricas
            })
            return {'subdivided': False, 'razon': razon}

        print(f"{'  ' * nivel}🔄 Subdividiendo cluster...")
        unique_labels = np.unique(clusterer.labels)
        
        for i, label in enumerate(unique_labels):
            subcluster_df = clusterer.data_df[clusterer.data_df['Cluster'] == label].copy()
            subcluster_df.drop(columns=['Cluster'], inplace=True)
            
            self.clustering_recursivo_memoria(
                (clusterer.header, subcluster_df), nivel + 1,
                min_atoms, max_iterations, n_clusters_target,
                silhouette_threshold, davies_bouldin_threshold,
                dispersion_threshold, quantile
            )

    def exportar_clusters_finales(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Exportando {len(self.final_clusters)} clusters finales a {output_dir}/")
        
        self.final_clusters.sort(key=lambda x: x['n_atoms'], reverse=True)
        
        for i, cluster_info in enumerate(self.final_clusters, 1):
            cluster_name = f"cluster_{i:03d}"
            output_file = output_path / f"{cluster_name}.dump"
            header, df = cluster_info['data_tuple']
            
            _write_lammps_dump(str(output_file), header, df)
            
            print(f"   ✓ {cluster_name}.dump - {cluster_info['n_atoms']} átomos (nivel {cluster_info['nivel']}) - {cluster_info['razon_final']}")
            
            cluster_info['nombre_final'] = cluster_name
            cluster_info['archivo_final'] = str(output_file)

def clustering_jerarquico_final(dump_file, output_dir="clusters_final",
                               min_atoms=50, max_iterations=5,
                               n_clusters_per_level=None,
                               silhouette_threshold=0.3,
                               davies_bouldin_threshold=1.5,
                               dispersion_threshold=None,
                               quantile=0.2,
                               limpiar_intermedios=True):
    print("="*70)
    print("  CLUSTERING JERÁRQUICO (SIN OVITO) - SOLO CLUSTERS FINALES")
    print("="*70)
    
    output_path = Path(output_dir)
    if limpiar_intermedios and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hierarchical_clusterer = HierarchicalMeanShiftClusterer()
    
    print(f"\n🔄 Iniciando clustering jerárquico iterativo...")
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
    
    resumen = {
        'archivo_original': dump_file,
        'n_clusters_finales': len(hierarchical_clusterer.final_clusters),
        'clusters_finales': []
    }
    for cluster in hierarchical_clusterer.final_clusters:
        resumen['clusters_finales'].append({
            'nombre': cluster.get('nombre_final', ''),
            'archivo': cluster.get('archivo_final', ''),
            'n_atoms': cluster.get('n_atoms', 0),
            'nivel': cluster.get('nivel', 0),
            'razon_final': cluster.get('razon_final', ''),
            'metricas': cluster.get('metricas', {})
        })
    
    json_path = output_path / "clustering_summary.json"
    with open(json_path, 'w') as f:
        json.dump(resumen, f, indent=2)
    
    print(f"\n📄 Resumen guardado en: {json_path}")
    print("\n✅ Proceso completado exitosamente!")
    return resumen

# ========================
# EJEMPLO DE USO
# ========================

if __name__ == "__main__":
    archivo_entrada = "dump.track_clustering"
    carpeta_salida = "clusters_finales_no_ovito"
    
    min_atomos = 12
    max_niveles = 15
    clusters_por_nivel = None
    
    umbral_silhouette = 0.9
    umbral_davies_bouldin = 1.5
    umbral_dispersion = 5.0
    
    if not Path(archivo_entrada).exists():
        print(f"❌ Error: El archivo de entrada '{archivo_entrada}' no fue encontrado.")
    else:
        resumen = clustering_jerarquico_final(
            dump_file=archivo_entrada,
            output_dir=carpeta_salida,
            min_atoms=min_atomos,
            max_iterations=max_niveles,
            n_clusters_per_level=clusters_por_nivel,
            silhouette_threshold=umbral_silhouette,
            davies_bouldin_threshold=umbral_davies_bouldin,
            dispersion_threshold=umbral_dispersion
        )
    # ========================
    # VERIFICACIÓN DE RESULTADOS
    # ========================
    
    print(f"\n🔍 Verificando integridad de los resultados...")
    
    # Contar átomos totales en archivos finales
    from pathlib import Path
    total_atoms_original = 0
    total_atoms_clusters = 0
    
    # Leer archivo original
    original = MeanShiftClusterer(archivo_entrada)
    total_atoms_original = original.leer_dump()
    
    # Sumar átomos en clusters finales
    for cluster_info in resumen['clusters_finales']:
        total_atoms_clusters += cluster_info['n_atoms']
    
    print(f"   • Átomos en archivo original: {total_atoms_original}")
    print(f"   • Átomos en clusters finales: {total_atoms_clusters}")
    
    if total_atoms_original == total_atoms_clusters:
        print(f"   ✅ Verificación exitosa: No se perdieron átomos")
    else:
        diferencia = total_atoms_original - total_atoms_clusters
        print(f"   ⚠️ ADVERTENCIA: Diferencia de {abs(diferencia)} átomos")
    
    print("\n" + "="*70)