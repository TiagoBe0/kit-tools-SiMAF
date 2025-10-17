#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VacancyPredict ML - Streamlit App CORREGIDA
Predicción de vacancias en redes cristalinas (lotes de archivos)
FUNCIONA: streamlit run vacancy_predict_fixed.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
from datetime import datetime
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, entropy
from scipy.ndimage import label
from sklearn.decomposition import PCA
from sklearn.cluster import estimate_bandwidth
from typing import List, Dict, Tuple
import zipfile
import io

A0 = 3.532
GRID_SIZE = (10, 10, 10)
BOX_SIZE_MAX = 10.0

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def parse_dump_file(dump_content: str) -> Tuple:
    """Parse LAMMPS dump file"""
    lines = dump_content.strip().split('\n')
    atoms_start = None
    n_atoms = None
    box_bounds = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('ITEM: TIMESTEP'):
            i += 1
            timestep = int(lines[i].strip())
        elif line.startswith('ITEM: NUMBER OF ATOMS'):
            i += 1
            n_atoms = int(lines[i].strip())
        elif line.startswith('ITEM: BOX BOUNDS'):
            i += 1
            box_bounds = []
            for _ in range(3):
                bounds = lines[i].strip().split()
                box_bounds.append([float(bounds[0]), float(bounds[1])])
                i += 1
            continue
        elif line.startswith('ITEM: ATOMS'):
            atoms_start = i + 1
            break
        i += 1
    
    if atoms_start is None or n_atoms is None:
        raise ValueError("Invalid dump format")
    
    positions = []
    for i in range(atoms_start, atoms_start + n_atoms):
        if i < len(lines):
            parts = lines[i].strip().split()
            if len(parts) >= 5:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                positions.append([x, y, z])
    
    positions = np.array(positions)
    if len(positions) == 0:
        raise ValueError("No atomic positions found")
    
    return positions, n_atoms, timestep, box_bounds

def normalize_positions(positions):
    """Normalize with optimized PCA"""
    if len(positions) < 3:
        return positions / A0, 2.0
    
    centered = positions - positions.mean(axis=0)
    try:
        pca = PCA(n_components=3, svd_solver='covariance_eigh')
        aligned = pca.fit_transform(centered)
    except:
        pca = PCA(n_components=3, svd_solver='auto')
        aligned = pca.fit_transform(centered)
    
    normalized = aligned / A0
    extent = normalized.max(axis=0) - normalized.min(axis=0)
    box_size = max(1.5, min(extent.max() * 1.5, BOX_SIZE_MAX))
    
    return normalized, box_size

def calc_grid_features(positions, box_size):
    """Calculate 26 grid features"""
    N, M, L = GRID_SIZE
    grid = np.zeros((N, M, L), dtype=np.int8)
    
    half_box = box_size / 2.0
    cell_size = box_size / np.array(GRID_SIZE)
    
    for pos in positions:
        indices = np.floor((pos + half_box) / cell_size).astype(int)
        if np.all(indices >= 0) and np.all(indices < GRID_SIZE):
            grid[indices[0], indices[1], indices[2]] = 1
    
    occupancy_total = grid.sum()
    features = {}
    
    features['occupancy_total'] = float(occupancy_total)
    features['occupancy_fraction'] = float(grid.mean())
    
    for axis, name in enumerate(['x', 'y', 'z']):
        slices = grid.sum(axis=axis)
        features[f'occupancy_{name}_mean'] = float(slices.mean())
    
    k_indices = np.arange(L)
    slices_z = grid.sum(axis=(0, 1))
    if occupancy_total > 0:
        com_k = (k_indices * slices_z).sum() / occupancy_total
        spread = np.sqrt(((k_indices - com_k)**2 * slices_z).sum() / occupancy_total)
        features['occupancy_spread_k'] = float(spread)
    else:
        features['occupancy_spread_k'] = 0.0
    
    if occupancy_total > 0:
        grad_x = np.abs(np.diff(grid, axis=0)).sum()
        grad_y = np.abs(np.diff(grid, axis=1)).sum()
        grad_z = np.abs(np.diff(grid, axis=2)).sum()
        features['occupancy_gradient_x'] = float(grad_x)
        features['occupancy_gradient_y'] = float(grad_y)
        features['occupancy_gradient_z'] = float(grad_z)
        features['occupancy_gradient_total'] = float(grad_x + grad_y + grad_z)
    else:
        features['occupancy_gradient_x'] = 0.0
        features['occupancy_gradient_y'] = 0.0
        features['occupancy_gradient_z'] = 0.0
        features['occupancy_gradient_total'] = 0.0
    
    try:
        labeled_grid, n_clusters = label(grid)
        features['n_fragments'] = int(n_clusters)
    except:
        features['n_fragments'] = 0
    
    if occupancy_total > 0:
        surface = (np.abs(np.diff(grid, axis=0)).sum() + 
                  np.abs(np.diff(grid, axis=1)).sum() + 
                  np.abs(np.diff(grid, axis=2)).sum())
        features['occupancy_surface'] = float(surface)
        features['occupancy_compactness'] = float(
            occupancy_total**(2/3) / (surface + 1e-8)
        )
    else:
        features['occupancy_surface'] = 0.0
        features['occupancy_compactness'] = 0.0
    
    if occupancy_total > 0:
        indices = np.argwhere(grid == 1)
        com = indices.mean(axis=0)
        features['grid_com_x'] = float(com[0] / N)
        features['grid_com_y'] = float(com[1] / M)
        features['grid_com_z'] = float(com[2] / L)
    else:
        features['grid_com_x'] = 0.5
        features['grid_com_y'] = 0.5
        features['grid_com_z'] = 0.5
    
    if occupancy_total > 0:
        x_profile = grid.sum(axis=(1, 2))
        x_indices = np.arange(N)
        x_mean = (x_indices * x_profile).sum() / occupancy_total
        x_centered = x_indices - x_mean
        skewness = (x_centered**3 * x_profile).sum() / (occupancy_total * (N/4)**3 + 1e-8)
        features['grid_skewness_x'] = float(skewness)
    else:
        features['grid_skewness_x'] = 0.0
    
    if occupancy_total > 0:
        prob = grid.flatten()
        prob = prob[prob > 0] / occupancy_total
        grid_entropy = -np.sum(prob * np.log(prob + 1e-10))
        features['grid_entropy'] = float(grid_entropy)
    else:
        features['grid_entropy'] = 0.0
    
    if occupancy_total > 0:
        try:
            coords = np.argwhere(grid == 1)
            centered = coords - coords.mean(axis=0)
            
            Ixx = np.sum(centered[:, 1]**2 + centered[:, 2]**2)
            Iyy = np.sum(centered[:, 0]**2 + centered[:, 2]**2)
            Izz = np.sum(centered[:, 0]**2 + centered[:, 1]**2)
            Ixy = -np.sum(centered[:, 0] * centered[:, 1])
            Ixz = -np.sum(centered[:, 0] * centered[:, 2])
            Iyz = -np.sum(centered[:, 1] * centered[:, 2])
            
            I_tensor = np.array([[Ixx, Ixy, Ixz],
                                [Ixy, Iyy, Iyz],
                                [Ixz, Iyz, Izz]])
            
            eigenvalues = np.sort(np.linalg.eigvalsh(I_tensor))[::-1]
            features['grid_moi_1'] = float(eigenvalues[0])
            features['grid_moi_2'] = float(eigenvalues[1])
            features['grid_moi_3'] = float(eigenvalues[2])
        except:
            features['grid_moi_1'] = 0.0
            features['grid_moi_2'] = 0.0
            features['grid_moi_3'] = 0.0
    else:
        features['grid_moi_1'] = 0.0
        features['grid_moi_2'] = 0.0
        features['grid_moi_3'] = 0.0
    
    for z in [0, 1, 4, 5, 8]:
        if z < L:
            features[f'occupancy_layer_z{z}'] = float(grid[:, :, z].sum())
        else:
            features[f'occupancy_layer_z{z}'] = 0.0
    
    return features

def calc_hull_features(positions):
    """Calculate ConvexHull features"""
    if len(positions) < 4:
        return {'hull_volume': np.nan, 'hull_area': np.nan}
    
    try:
        hull = ConvexHull(positions)
        return {
            'hull_volume': float(hull.volume),
            'hull_area': float(hull.area)
        }
    except:
        return {'hull_volume': np.nan, 'hull_area': np.nan}

def calc_inertia_features(positions):
    """Calculate principal moments of inertia"""
    if len(positions) < 3:
        return {f'moi_principal_{i}': np.nan for i in [1, 2, 3]}
    
    centered = positions - positions.mean(axis=0)
    
    Ixx = np.sum(centered[:, 1]**2 + centered[:, 2]**2)
    Iyy = np.sum(centered[:, 0]**2 + centered[:, 2]**2)
    Izz = np.sum(centered[:, 0]**2 + centered[:, 1]**2)
    
    Ixy = -np.sum(centered[:, 0] * centered[:, 1])
    Ixz = -np.sum(centered[:, 0] * centered[:, 2])
    Iyz = -np.sum(centered[:, 1] * centered[:, 2])
    
    I_tensor = np.array([[Ixx, Ixy, Ixz],
                        [Ixy, Iyy, Iyz],
                        [Ixz, Iyz, Izz]])
    
    eigenvalues = np.sort(np.linalg.eigvalsh(I_tensor))[::-1]
    
    return {
        'moi_principal_1': float(eigenvalues[0]),
        'moi_principal_2': float(eigenvalues[1]),
        'moi_principal_3': float(eigenvalues[2])
    }

def calc_radial_features(positions):
    """Calculate radial distribution"""
    if len(positions) < 2:
        return {'rdf_mean': np.nan, 'rdf_kurtosis': np.nan}
    
    com = positions.mean(axis=0)
    distances = np.linalg.norm(positions - com, axis=1)
    
    return {
        'rdf_mean': float(distances.mean()),
        'rdf_kurtosis': float(kurtosis(distances))
    }

def calc_entropy_feature(positions):
    """Calculate spatial entropy"""
    if len(positions) < 2:
        return {'entropy_spatial': np.nan}
    
    H, _ = np.histogramdd(positions, bins=10)
    H_flat = H.flatten()
    H_norm = H_flat[H_flat > 0] / H_flat.sum()
    
    return {'entropy_spatial': float(entropy(H_norm))}

def calc_bandwidth_feature(positions):
    """Calculate clustering bandwidth"""
    if len(positions) < 10:
        return {'ms_bandwidth': np.nan}
    
    try:
        bandwidth = estimate_bandwidth(
            positions, 
            quantile=0.2, 
            n_samples=min(500, len(positions))
        )
        
        if bandwidth <= 0:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(positions)
            distances, _ = nn.kneighbors(positions)
            bandwidth = np.mean(distances[:, 1]) * 2.0
        
        return {'ms_bandwidth': float(bandwidth)}
    except:
        return {'ms_bandwidth': np.nan}

def extract_features(positions):
    """Extract all 37 features"""
    normalized_pos, box_size = normalize_positions(positions)
    
    features = {}
    features.update(calc_grid_features(normalized_pos, box_size))
    features.update(calc_hull_features(positions))
    features.update(calc_inertia_features(positions))
    features.update(calc_radial_features(positions))
    features.update(calc_entropy_feature(positions))
    features.update(calc_bandwidth_feature(positions))
    
    return features

# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_single_file(file_content: str, filename: str, model) -> Dict:
    """Procesa un único archivo dump"""
    try:
        positions, n_atoms, timestep, box_bounds = parse_dump_file(file_content)
        features = extract_features(positions)
        X_pred = pd.DataFrame([features])
        prediction = model.predict(X_pred)[0]
        
        return {
            'archivo': filename,
            'atoms': n_atoms,
            'timestep': timestep,
            'prediction': float(prediction),
            'status': 'success',
            'error': None
        }
    except Exception as e:
        return {
            'archivo': filename,
            'atoms': None,
            'timestep': None,
            'prediction': None,
            'status': 'error',
            'error': str(e)
        }

def process_batch_files(files: List, model) -> Tuple[pd.DataFrame, Dict]:
    """Procesa lote de archivos y retorna resultados"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(files):
        try:
            # Actualizar progreso
            progress = (idx + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text(f"Procesando: {file.name} ({idx + 1}/{len(files)})")
            
            # Leer contenido
            file_content = file.read().decode('utf-8')
            file.seek(0)
            
            # Procesar
            result = process_single_file(file_content, file.name, model)
            results.append(result)
            
        except Exception as e:
            results.append({
                'archivo': file.name,
                'atoms': None,
                'timestep': None,
                'prediction': None,
                'status': 'error',
                'error': f"Critical: {str(e)}"
            })
    
    progress_bar.empty()
    status_text.empty()
    
    df_results = pd.DataFrame(results)
    
    # Estadísticas
    successful = len(df_results[df_results['status'] == 'success'])
    failed = len(df_results[df_results['status'] == 'error'])
    
    stats = {
        'total_files': len(files),
        'successful': successful,
        'failed': failed,
        'mean_prediction': df_results[df_results['status'] == 'success']['prediction'].mean() if successful > 0 else 0,
        'std_prediction': df_results[df_results['status'] == 'success']['prediction'].std() if successful > 0 else 0,
        'min_prediction': df_results[df_results['status'] == 'success']['prediction'].min() if successful > 0 else 0,
        'max_prediction': df_results[df_results['status'] == 'success']['prediction'].max() if successful > 0 else 0,
    }
    
    return df_results, stats

def export_results_csv(df_results: pd.DataFrame) -> bytes:
    """Exporta resultados a CSV"""
    return df_results.to_csv(index=False).encode('utf-8')

def export_results_zip(df_results: pd.DataFrame, stats: Dict) -> bytes:
    """Exporta resultados a ZIP"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        csv_data = df_results.to_csv(index=False)
        zf.writestr('resultados.csv', csv_data)
        
        json_data = json.dumps(stats, indent=2)
        zf.writestr('resumen.json', json_data)
        
        timestamp_data = f"Procesamiento: {datetime.now().isoformat()}\n"
        zf.writestr('metadata.txt', timestamp_data)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="VacancyPredict ML - Batch",
    page_icon="🔬",
    layout="wide"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #667eea;'>VacancyPredict ML</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Predicción de vacancias - Procesamiento en Lotes</p>", 
                unsafe_allow_html=True)

st.divider()

# Sidebar - Cargar modelo
with st.sidebar:
    st.markdown("# 📦 Cargar Modelo")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    available_models = sorted([f for f in models_dir.glob("*.pkl")])
    
    if available_models:
        selected_model = st.selectbox(
            "Selecciona modelo pre-entrenado",
            available_models,
            format_func=lambda x: x.name
        )
        
        if st.button("✅ Cargar Modelo", use_container_width=True):
            try:
                model = joblib.load(selected_model)
                st.success(f"✓ Modelo cargado: {selected_model.name}")
                st.session_state.model = model
                st.session_state.model_name = selected_model.name
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.info("No hay modelos en 'models/'")
    
    st.divider()
    
    st.markdown("### O carga manualmente")
    model_file = st.file_uploader("Selecciona (.pkl)", type=['pkl'])
    
    if model_file:
        try:
            model = joblib.load(model_file)
            st.success("✓ Modelo cargado")
            st.session_state.model = model
            st.session_state.model_name = model_file.name
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.divider()
    if 'model' in st.session_state:
        st.markdown("### 🎯 Modelo Activo")
        st.info(f"**{st.session_state.model_name}**")
    else:
        st.warning("⚠️ Carga un modelo")

# Main content
st.markdown("## 📋 Procesamiento de Lotes")

if 'model' not in st.session_state:
    st.warning("⚠️ Por favor carga un modelo en la barra lateral")
else:
    st.markdown("### Selecciona múltiples archivos")
    uploaded_files = st.file_uploader(
        "Arrastra archivos .dump o .txt",
        type=['dump', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**📁 Archivos cargados: {len(uploaded_files)}**")
        
        with st.expander("Ver archivos cargados", expanded=False):
            for i, f in enumerate(uploaded_files, 1):
                st.text(f"{i}. {f.name}")
        
        if st.button("🚀 Procesar Lote", use_container_width=True):
            st.markdown("---")
            
            with st.spinner("Procesando archivos..."):
                df_results, stats = process_batch_files(
                    uploaded_files,
                    st.session_state.model
                )
            
            # Resultados
            st.markdown("## ✅ Resultados del Procesamiento")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Total procesado", stats['total_files'])
            col2.metric("✓ Exitosos", stats['successful'])
            col3.metric("✗ Errores", stats['failed'])
            col4.metric("⏱ Promedio vacancias", f"{stats['mean_prediction']:.2f}")
            
            st.divider()
            
            st.markdown("### Tabla de Resultados")
            st.dataframe(df_results, use_container_width=True)
            
            st.divider()
            
            if stats['successful'] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Estadísticas")
                    st.metric("Predicción Mínima", f"{stats['min_prediction']:.2f}")
                    st.metric("Predicción Máxima", f"{stats['max_prediction']:.2f}")
                
                with col2:
                    st.markdown("### 📊 Desviación")
                    st.metric("Promedio", f"{stats['mean_prediction']:.2f}")
                    st.metric("Std Dev", f"{stats['std_prediction']:.2f}")
            
            st.divider()
            
            st.markdown("### 💾 Descargar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = export_results_csv(df_results)
                st.download_button(
                    label="📥 CSV",
                    data=csv_data,
                    file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                zip_data = export_results_zip(df_results, stats)
                st.download_button(
                    label="📦 ZIP (CSV + JSON)",
                    data=zip_data,
                    file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            st.session_state.last_results = df_results
            st.session_state.last_stats = stats

st.divider()
st.markdown("""
<div style='text-align: center; color: #999;'>
VacancyPredict ML v2.0 | Batch Processing | Predicción de Vacancias
</div>
""", unsafe_allow_html=True)