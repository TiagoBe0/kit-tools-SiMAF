#!/usr/bin/env python3
"""
Interfaz Streamlit para Análisis Wigner-Seitz
Versión con Plotly - Gráficos 3D interactivos
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from pathlib import Path
from scipy.spatial import cKDTree

# Importar las clases del script original
from wigner_seitz_strain import (
    SimulationBox,
    WignerSeitzAnalyzer,
    read_lammps_dump
)

# Configuración de la página
st.set_page_config(
    page_title="Análisis Wigner-Seitz",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def create_3d_scatter_interactive(analyzer, results, plot_range, show_normal, show_vacancies, show_interstitials, 
                                   use_auto_range=True, ref_box=None):
    """Visualización 3D interactiva con Plotly
    
    Args:
        analyzer: WignerSeitzAnalyzer instance
        results: Diccionario de resultados
        plot_range: Rango manual (usado si use_auto_range=False)
        show_normal: Mostrar átomos normales
        show_vacancies: Mostrar vacancias
        show_interstitials: Mostrar intersticiales
        use_auto_range: Si True, calcula el rango basado en la celda de simulación
        ref_box: SimulationBox de referencia (necesario si use_auto_range=True)
    """
    fig = go.Figure()
    
    # Calcular rangos basados en la celda de simulación
    if use_auto_range and ref_box is not None:
        # Obtener las dimensiones de la caja desde SimulationBox
        x_range = [ref_box.xlo, ref_box.xhi]
        y_range = [ref_box.ylo, ref_box.yhi]
        z_range = [ref_box.zlo, ref_box.zhi]
    else:
        # Usar rango manual
        x_range = [-plot_range, plot_range]
        y_range = [-plot_range, plot_range]
        z_range = [-plot_range, plot_range]
    
    # Sitios normales
    if show_normal:
        normal = np.where(analyzer.occupancy == 1)[0]
        if len(normal) > 0:
            pos = analyzer.reference[normal]
            # Filtrar puntos dentro del rango
            if use_auto_range:
                mask = ((pos[:, 0] >= x_range[0]) & (pos[:, 0] <= x_range[1]) &
                       (pos[:, 1] >= y_range[0]) & (pos[:, 1] <= y_range[1]) &
                       (pos[:, 2] >= z_range[0]) & (pos[:, 2] <= z_range[1]))
            else:
                mask = np.all(np.abs(pos) < plot_range, axis=1)
            pos = pos[mask]
            if len(pos) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pos[:, 0],
                    y=pos[:, 1],
                    z=pos[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='lightblue',
                        opacity=0.3,
                    ),
                    name='Normal',
                    hovertemplate='<b>Normal</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                ))
    
    # Vacancias
    if show_vacancies and len(results['vacancies']) > 0:
        pos = analyzer.reference[results['vacancies']]
        if use_auto_range:
            mask = ((pos[:, 0] >= x_range[0]) & (pos[:, 0] <= x_range[1]) &
                   (pos[:, 1] >= y_range[0]) & (pos[:, 1] <= y_range[1]) &
                   (pos[:, 2] >= z_range[0]) & (pos[:, 2] <= z_range[1]))
        else:
            mask = np.all(np.abs(pos) < plot_range, axis=1)
        pos = pos[mask]
        if len(pos) > 0:
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                name=f'Vacancias ({len(results["vacancies"])})',
                hovertemplate='<b>Vacancia</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
    
    # Intersticiales
    if show_interstitials and len(results['interstitial_atoms']) > 0:
        pos = analyzer.defective[results['interstitial_atoms']]
        if use_auto_range:
            mask = ((pos[:, 0] >= x_range[0]) & (pos[:, 0] <= x_range[1]) &
                   (pos[:, 1] >= y_range[0]) & (pos[:, 1] <= y_range[1]) &
                   (pos[:, 2] >= z_range[0]) & (pos[:, 2] <= z_range[1]))
        else:
            mask = np.all(np.abs(pos) < plot_range, axis=1)
        pos = pos[mask]
        if len(pos) > 0:
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color='orange',
                    symbol='diamond',
                    line=dict(width=1, color='darkorange')
                ),
                name=f'Intersticiales ({len(results["interstitial_atoms"])})',
                hovertemplate='<b>Intersticial</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ))
    
    # Configuración del layout
    title = 'Análisis Wigner-Seitz - Visualización Interactiva'
    if analyzer.use_affine_mapping:
        title += ' (con mapeo afín)'
    
    # Calcular aspect ratio para mantener proporciones correctas
    if use_auto_range:
        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]
        z_size = z_range[1] - z_range[0]
        max_size = max(x_size, y_size, z_size)
        aspect_ratio = dict(
            x=x_size/max_size,
            y=y_size/max_size,
            z=z_size/max_size
        )
        aspectmode = 'manual'
    else:
        aspect_ratio = dict(x=1, y=1, z=1)
        aspectmode = 'cube'
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1f77b4')),
        scene=dict(
            xaxis=dict(title='X (Å)', range=x_range, backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='Y (Å)', range=y_range, backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title='Z (Å)', range=z_range, backgroundcolor="rgb(230, 230,230)"),
            aspectmode=aspectmode,
            aspectratio=aspect_ratio
        ),
        height=700,
        showlegend=True,
        legend=dict(x=0.7, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest'
    )
    
    return fig


def create_comparison_interactive(analyzer, results, plot_range):
    """Comparación lado a lado interactiva"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Estructura de Referencia<br>(con vacancias)', 
                       'Estructura Defectuosa<br>(con intersticiales)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )
    
    # Panel izquierdo: Referencia
    pos_ref = analyzer.reference
    mask = np.all(np.abs(pos_ref) < plot_range, axis=1)
    pos_ref_filtered = pos_ref[mask]
    
    if len(pos_ref_filtered) > 0:
        fig.add_trace(go.Scatter3d(
            x=pos_ref_filtered[:, 0],
            y=pos_ref_filtered[:, 1],
            z=pos_ref_filtered[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.4),
            name='Sitios ref.',
            hovertemplate='<b>Sitio</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            showlegend=True
        ), row=1, col=1)
    
    # Vacancias en referencia
    if len(results['vacancies']) > 0:
        pos_vac = analyzer.reference[results['vacancies']]
        mask_vac = np.all(np.abs(pos_vac) < plot_range, axis=1)
        pos_vac_filtered = pos_vac[mask_vac]
        if len(pos_vac_filtered) > 0:
            fig.add_trace(go.Scatter3d(
                x=pos_vac_filtered[:, 0],
                y=pos_vac_filtered[:, 1],
                z=pos_vac_filtered[:, 2],
                mode='markers',
                marker=dict(size=8, color='red', symbol='x', line=dict(width=2, color='darkred')),
                name='Vacancias',
                hovertemplate='<b>Vacancia</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                showlegend=True
            ), row=1, col=1)
    
    # Panel derecho: Defectuosa
    pos_def = analyzer.defective
    mask_def = np.all(np.abs(pos_def) < plot_range, axis=1)
    
    # Átomos normales
    normal_atoms = np.ones(len(analyzer.defective), dtype=bool)
    normal_atoms[results['interstitial_atoms']] = False
    
    pos_normal = pos_def[normal_atoms]
    mask_normal = np.all(np.abs(pos_normal) < plot_range, axis=1)
    pos_normal_filtered = pos_normal[mask_normal]
    
    if len(pos_normal_filtered) > 0:
        fig.add_trace(go.Scatter3d(
            x=pos_normal_filtered[:, 0],
            y=pos_normal_filtered[:, 1],
            z=pos_normal_filtered[:, 2],
            mode='markers',
            marker=dict(size=3, color='green', opacity=0.4),
            name='Átomos norm.',
            hovertemplate='<b>Átomo</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            showlegend=True
        ), row=1, col=2)
    
    # Intersticiales
    if len(results['interstitial_atoms']) > 0:
        pos_int = analyzer.defective[results['interstitial_atoms']]
        mask_int = np.all(np.abs(pos_int) < plot_range, axis=1)
        pos_int_filtered = pos_int[mask_int]
        if len(pos_int_filtered) > 0:
            fig.add_trace(go.Scatter3d(
                x=pos_int_filtered[:, 0],
                y=pos_int_filtered[:, 1],
                z=pos_int_filtered[:, 2],
                mode='markers',
                marker=dict(size=8, color='orange', symbol='diamond', line=dict(width=1, color='darkorange')),
                name='Intersticiales',
                hovertemplate='<b>Intersticial</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                showlegend=True
            ), row=1, col=2)
    
    # Configuración del layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Comparación Referencia vs Defectuosa",
        title_font_size=18
    )
    
    # Configurar ambos ejes 3D
    for col in [1, 2]:
        fig.update_scenes(
            xaxis=dict(title='X (Å)', range=[-plot_range, plot_range]),
            yaxis=dict(title='Y (Å)', range=[-plot_range, plot_range]),
            zaxis=dict(title='Z (Å)', range=[-plot_range, plot_range]),
            aspectmode='cube',
            row=1, col=col
        )
    
    return fig


def create_density_interactive(analyzer, results, plot_range):
    """Histogramas de densidad 3D interactivos"""
    
    # Preparar datos
    if len(results['vacancies']) > 0:
        vac_pos = analyzer.reference[results['vacancies']]
        mask_vac = np.all(np.abs(vac_pos) < plot_range, axis=1)
        vac_pos = vac_pos[mask_vac]
    else:
        vac_pos = np.array([]).reshape(0, 3)
    
    if len(results['interstitial_atoms']) > 0:
        int_pos = analyzer.defective[results['interstitial_atoms']]
        mask_int = np.all(np.abs(int_pos) < plot_range, axis=1)
        int_pos = int_pos[mask_int]
    else:
        int_pos = np.array([]).reshape(0, 3)
    
    # Crear subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Distribución de Vacancias', 
                       'Distribución de Intersticiales',
                       'Defectos Combinados'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )
    
    # Panel 1: Vacancias con densidad
    if len(vac_pos) > 0:
        hist, edges = np.histogramdd(vac_pos, bins=15, range=[(-plot_range, plot_range)]*3)
        nonzero = np.where(hist > 0)
        
        x_centers = (edges[0][:-1] + edges[0][1:]) / 2
        y_centers = (edges[1][:-1] + edges[1][1:]) / 2
        z_centers = (edges[2][:-1] + edges[2][1:]) / 2
        
        x = x_centers[nonzero[0]]
        y = y_centers[nonzero[1]]
        z = z_centers[nonzero[2]]
        counts = hist[nonzero]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=counts * 5,
                color=counts,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(x=0.28, len=0.5, title="Densidad"),
                opacity=0.7
            ),
            name='Densidad Vac.',
            hovertemplate='<b>Bin</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>Count: %{marker.color}<extra></extra>'
        ), row=1, col=1)
    
    # Panel 2: Intersticiales con densidad
    if len(int_pos) > 0:
        hist, edges = np.histogramdd(int_pos, bins=15, range=[(-plot_range, plot_range)]*3)
        nonzero = np.where(hist > 0)
        
        x_centers = (edges[0][:-1] + edges[0][1:]) / 2
        y_centers = (edges[1][:-1] + edges[1][1:]) / 2
        z_centers = (edges[2][:-1] + edges[2][1:]) / 2
        
        x = x_centers[nonzero[0]]
        y = y_centers[nonzero[1]]
        z = z_centers[nonzero[2]]
        counts = hist[nonzero]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=counts * 5,
                color=counts,
                colorscale='Oranges',
                showscale=True,
                colorbar=dict(x=0.63, len=0.5, title="Densidad"),
                opacity=0.7
            ),
            name='Densidad Int.',
            hovertemplate='<b>Bin</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>Count: %{marker.color}<extra></extra>'
        ), row=1, col=2)
    
    # Panel 3: Combinado
    if len(vac_pos) > 0:
        fig.add_trace(go.Scatter3d(
            x=vac_pos[:, 0],
            y=vac_pos[:, 1],
            z=vac_pos[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.7),
            name='Vacancias',
            hovertemplate='<b>Vacancia</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ), row=1, col=3)
    
    if len(int_pos) > 0:
        fig.add_trace(go.Scatter3d(
            x=int_pos[:, 0],
            y=int_pos[:, 1],
            z=int_pos[:, 2],
            mode='markers',
            marker=dict(size=5, color='orange', symbol='diamond', opacity=0.7),
            name='Intersticiales',
            hovertemplate='<b>Intersticial</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ), row=1, col=3)
    
    # Configurar layout
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Análisis de Densidad Espacial",
        title_font_size=18
    )
    
    # Configurar ejes 3D para todos los paneles
    for col in [1, 2, 3]:
        fig.update_scenes(
            xaxis=dict(title='X (Å)'),
            yaxis=dict(title='Y (Å)'),
            zaxis=dict(title='Z (Å)'),
            aspectmode='cube',
            row=1, col=col
        )
    
    return fig


def create_slices_interactive(analyzer, results, plot_range, slice_thickness=5.0):
    """Vistas de planos de corte interactivas"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Plano XY (Z≈0)', 'Plano XZ (Y≈0)', 'Plano YZ (X≈0)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    planes = [
        ('XY', [0, 1], 2, 1),
        ('XZ', [0, 2], 1, 2),
        ('YZ', [1, 2], 0, 3)
    ]
    
    for plane_name, plot_dims, slice_dim, col in planes:
        # Filtrar puntos
        mask_ref = np.abs(analyzer.reference[:, slice_dim]) < slice_thickness
        
        # Sitios normales
        normal = np.where((analyzer.occupancy == 1) & mask_ref)[0]
        if len(normal) > 0:
            pos = analyzer.reference[normal][:, plot_dims]
            mask_range = np.all(np.abs(pos) < plot_range, axis=1)
            pos = pos[mask_range]
            if len(pos) > 0:
                fig.add_trace(go.Scatter(
                    x=pos[:, 0],
                    y=pos[:, 1],
                    mode='markers',
                    marker=dict(size=4, color='lightblue', opacity=0.4),
                    name='Normal',
                    showlegend=(col == 1),
                    hovertemplate='<b>Normal</b><br>Pos: (%{x:.2f}, %{y:.2f})<extra></extra>'
                ), row=1, col=col)
        
        # Vacancias
        if len(results['vacancies']) > 0:
            vac_mask = np.isin(np.arange(len(analyzer.reference)), results['vacancies']) & mask_ref
            if np.any(vac_mask):
                pos = analyzer.reference[vac_mask][:, plot_dims]
                mask_range = np.all(np.abs(pos) < plot_range, axis=1)
                pos = pos[mask_range]
                if len(pos) > 0:
                    fig.add_trace(go.Scatter(
                        x=pos[:, 0],
                        y=pos[:, 1],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='x', line=dict(width=2)),
                        name='Vacancias',
                        showlegend=(col == 1),
                        hovertemplate='<b>Vacancia</b><br>Pos: (%{x:.2f}, %{y:.2f})<extra></extra>'
                    ), row=1, col=col)
        
        # Intersticiales
        if len(results['interstitial_atoms']) > 0:
            mask_def = np.abs(analyzer.defective[:, slice_dim]) < slice_thickness
            int_mask = np.isin(np.arange(len(analyzer.defective)), results['interstitial_atoms']) & mask_def
            if np.any(int_mask):
                pos = analyzer.defective[int_mask][:, plot_dims]
                mask_range = np.all(np.abs(pos) < plot_range, axis=1)
                pos = pos[mask_range]
                if len(pos) > 0:
                    fig.add_trace(go.Scatter(
                        x=pos[:, 0],
                        y=pos[:, 1],
                        mode='markers',
                        marker=dict(size=10, color='orange', symbol='diamond'),
                        name='Intersticiales',
                        showlegend=(col == 1),
                        hovertemplate='<b>Intersticial</b><br>Pos: (%{x:.2f}, %{y:.2f})<extra></extra>'
                    ), row=1, col=col)
        
        # Configurar ejes
        axis_labels = ['X (Å)', 'Y (Å)', 'Z (Å)']
        fig.update_xaxes(title_text=axis_labels[plot_dims[0]], range=[-plot_range, plot_range], 
                        row=1, col=col, scaleanchor=f"y{col}", scaleratio=1)
        fig.update_yaxes(title_text=axis_labels[plot_dims[1]], range=[-plot_range, plot_range], 
                        row=1, col=col)
    
    fig.update_layout(
        height=500,
        title_text=f"Vistas en Planos de Corte (espesor ±{slice_thickness} Å)",
        title_font_size=18,
        showlegend=True
    )
    
    return fig


def create_nearest_neighbor_plot(analyzer, results, plot_range):
    """Análisis de distancias entre defectos"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Distancias Vacancia-Vacancia', 
                       'Distancias Intersticial-Intersticial',
                       'Distancias Vacancia-Intersticial')
    )
    
    # Obtener posiciones
    if len(results['vacancies']) > 0:
        vac_pos = analyzer.reference[results['vacancies']]
        mask_vac = np.all(np.abs(vac_pos) < plot_range, axis=1)
        vac_pos = vac_pos[mask_vac]
    else:
        vac_pos = np.array([]).reshape(0, 3)
    
    if len(results['interstitial_atoms']) > 0:
        int_pos = analyzer.defective[results['interstitial_atoms']]
        mask_int = np.all(np.abs(int_pos) < plot_range, axis=1)
        int_pos = int_pos[mask_int]
    else:
        int_pos = np.array([]).reshape(0, 3)
    
    # Panel 1: Vacancias
    if len(vac_pos) > 1:
        tree = cKDTree(vac_pos)
        distances, _ = tree.query(vac_pos, k=2)
        nn_distances = distances[:, 1]
        
        fig.add_trace(go.Histogram(
            x=nn_distances,
            nbinsx=30,
            marker_color='red',
            opacity=0.7,
            name='Vac-Vac',
            hovertemplate='Distancia: %{x:.2f} Å<br>Frecuencia: %{y}<extra></extra>'
        ), row=1, col=1)
        
        # Línea de promedio
        mean_dist = np.mean(nn_distances)
        fig.add_vline(x=mean_dist, line_dash="dash", line_color="darkred", 
                     annotation_text=f"μ={mean_dist:.2f}Å", row=1, col=1)
    
    # Panel 2: Intersticiales
    if len(int_pos) > 1:
        tree = cKDTree(int_pos)
        distances, _ = tree.query(int_pos, k=2)
        nn_distances = distances[:, 1]
        
        fig.add_trace(go.Histogram(
            x=nn_distances,
            nbinsx=30,
            marker_color='orange',
            opacity=0.7,
            name='Int-Int',
            hovertemplate='Distancia: %{x:.2f} Å<br>Frecuencia: %{y}<extra></extra>'
        ), row=1, col=2)
        
        mean_dist = np.mean(nn_distances)
        fig.add_vline(x=mean_dist, line_dash="dash", line_color="darkorange",
                     annotation_text=f"μ={mean_dist:.2f}Å", row=1, col=2)
    
    # Panel 3: Vacancia-Intersticial
    if len(vac_pos) > 0 and len(int_pos) > 0:
        tree_vac = cKDTree(vac_pos)
        distances, _ = tree_vac.query(int_pos)
        
        fig.add_trace(go.Histogram(
            x=distances,
            nbinsx=30,
            marker_color='purple',
            opacity=0.7,
            name='Vac-Int',
            hovertemplate='Distancia: %{x:.2f} Å<br>Frecuencia: %{y}<extra></extra>'
        ), row=1, col=3)
        
        mean_dist = np.mean(distances)
        fig.add_vline(x=mean_dist, line_dash="dash", line_color="darkviolet",
                     annotation_text=f"μ={mean_dist:.2f}Å", row=1, col=3)
    
    # Configurar layout
    fig.update_xaxes(title_text="Distancia (Å)")
    fig.update_yaxes(title_text="Frecuencia")
    
    fig.update_layout(
        height=500,
        title_text="Análisis de Vecinos Cercanos",
        title_font_size=18,
        showlegend=False
    )
    
    return fig
def export_defects_dump(analyzer, results, filename="defects_only.dump"):
    """
    Exporta un archivo LAMMPS .dump con las vacancias e intersticiales detectadas.
    """
    vac_pos = analyzer.reference[results['vacancies']] if len(results['vacancies']) > 0 else np.empty((0,3))
    int_pos = analyzer.defective[results['interstitial_atoms']] if len(results['interstitial_atoms']) > 0 else np.empty((0,3))

    # Concatenar posiciones y asignar tipo: 1 = vacancia, 2 = intersticial
    all_pos = np.vstack((vac_pos, int_pos))
    types = np.array([1]*len(vac_pos) + [2]*len(int_pos))

    box = analyzer.reference_box  # <- corrección clave

    with open(filename, "w") as f:
        f.write("ITEM: TIMESTEP\n0\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{len(all_pos)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{box.xlo} {box.xhi}\n")
        f.write(f"{box.ylo} {box.yhi}\n")
        f.write(f"{box.zlo} {box.zhi}\n")
        f.write("ITEM: ATOMS id type x y z\n")

        for i, (typ, pos) in enumerate(zip(types, all_pos), start=1):
            f.write(f"{i} {typ} {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}\n")

    return filename



def main():
    # Título principal
    st.markdown('<div class="main-header">🔬 Análisis Wigner-Seitz Interactive</div>', 
                unsafe_allow_html=True)
    st.markdown("### Visualizaciones 3D interactivas con Plotly - Rota, zoom y explora")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
    with st.sidebar.expander("ℹ️ Acerca del método"):
        st.markdown("""
        **Análisis Wigner-Seitz:**
        - Detecta vacancias (sitios vacíos)
        - Identifica intersticiales (átomos extras)
        - Soporta strain > 5% mediante mapeo afín
        - Condiciones periódicas de contorno (PBC)
        
        **Gráficos interactivos:**
        - 🖱️ Clic y arrastra para rotar
        - 🔍 Scroll para zoom
        - 📍 Hover para ver coordenadas
        - 📷 Botón de cámara para guardar
        """)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Cargar Archivos", "🔍 Análisis", 
                                       "📊 Resultados", "🎨 Visualizaciones"])
    
    # ========================================================================
    # TAB 1: CARGAR ARCHIVOS
    # ========================================================================
    with tab1:
        st.header("Cargar archivos LAMMPS dump")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔹 Configuración de Referencia")
            ref_file = st.file_uploader(
                "Archivo de referencia (estructura perfecta)",
                type=['dump', 'txt'],
                key='ref_file',
                help="Estructura cristalina perfecta sin defectos"
            )
            
            if ref_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dump') as tmp:
                    tmp.write(ref_file.getvalue())
                    ref_path = tmp.name
                
                try:
                    ref_positions, ref_box = read_lammps_dump(ref_path)
                    st.session_state['ref_positions'] = ref_positions
                    st.session_state['ref_box'] = ref_box
                    
                    st.success(f"✅ Cargado: {len(ref_positions)} sitios")
                    
                    with st.expander("Ver detalles de la caja"):
                        st.write(f"**Dimensiones:**")
                        st.write(f"- Lx: {ref_box.lx:.3f} Å")
                        st.write(f"- Ly: {ref_box.ly:.3f} Å")
                        st.write(f"- Lz: {ref_box.lz:.3f} Å")
                        st.write(f"**Volumen:** {ref_box.get_volume():.3f} Ų")
                        
                except Exception as e:
                    st.error(f"❌ Error al cargar: {str(e)}")
                finally:
                    os.unlink(ref_path)
        
        with col2:
            st.subheader("🔸 Configuración Defectuosa")
            def_file = st.file_uploader(
                "Archivo defectuoso (con vacancias/intersticiales)",
                type=['dump', 'txt'],
                key='def_file',
                help="Estructura con defectos cristalinos"
            )
            
            if def_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dump') as tmp:
                    tmp.write(def_file.getvalue())
                    def_path = tmp.name
                
                try:
                    def_positions, def_box = read_lammps_dump(def_path)
                    st.session_state['def_positions'] = def_positions
                    st.session_state['def_box'] = def_box
                    
                    st.success(f"✅ Cargado: {len(def_positions)} átomos")
                    
                    with st.expander("Ver detalles de la caja"):
                        st.write(f"**Dimensiones:**")
                        st.write(f"- Lx: {def_box.lx:.3f} Å")
                        st.write(f"- Ly: {def_box.ly:.3f} Å")
                        st.write(f"- Lz: {def_box.lz:.3f} Å")
                        st.write(f"**Volumen:** {def_box.get_volume():.3f} Ų")
                        
                        if 'ref_box' in st.session_state:
                            strain = def_box.get_strain(st.session_state['ref_box'])
                            st.write(f"**Strain volumétrico:** {strain*100:.2f}%")
                            
                            if abs(strain) > 0.05:
                                st.warning("⚠️ Strain > 5%: Se recomienda activar mapeo afín")
                        
                except Exception as e:
                    st.error(f"❌ Error al cargar: {str(e)}")
                finally:
                    os.unlink(def_path)
        
        if 'ref_positions' in st.session_state and 'def_positions' in st.session_state:
            st.success("✅ Ambos archivos cargados. Continúa al 'Análisis'")
        else:
            st.info("👆 Carga ambos archivos para comenzar")
    
    # ========================================================================
    # TAB 2: ANÁLISIS
    # ========================================================================
    with tab2:
        st.header("Configuración del Análisis")
        
        if 'ref_positions' not in st.session_state or 'def_positions' not in st.session_state:
            st.warning("⚠️ Primero carga ambos archivos")
            return
        
        st.subheader("Parámetros")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_pbc = st.checkbox(
                "Condiciones periódicas (PBC)",
                value=True,
                help="Activar/desactivar PBC"
            )
        
        with col2:
            use_affine = st.checkbox(
                "Mapeo afín (strain > 5%)",
                value=False,
                help="Corrige deformaciones grandes"
            )
        
        if 'ref_box' in st.session_state and 'def_box' in st.session_state:
            strain = st.session_state['def_box'].get_strain(st.session_state['ref_box'])
            
            if abs(strain) > 0.05 and not use_affine:
                st.error(f"""
                ⚠️ **ADVERTENCIA:** Strain = {strain*100:.2f}%
                
                Activa el **mapeo afín** para strain > 5%.
                """)
            elif abs(strain) > 0.05 and use_affine:
                st.success(f"✅ Mapeo afín activado (strain = {strain*100:.2f}%)")
        
        st.markdown("---")
        
        if st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
            with st.spinner("Analizando..."):
                try:
                    analyzer = WignerSeitzAnalyzer(
                        st.session_state['ref_positions'],
                        st.session_state['def_positions'],
                        st.session_state['ref_box'],
                        st.session_state['def_box'],
                        use_pbc=use_pbc,
                        use_affine_mapping=use_affine
                    )
                    
                    results = analyzer.analyze()
                    
                    st.session_state['results'] = results
                    st.session_state['analyzer'] = analyzer
                    
                    st.success("✅ Análisis completado!")
                    st.info("📊 Ve a las pestañas 'Resultados' y 'Visualizaciones'")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # ========================================================================
    # TAB 3: RESULTADOS
    # ========================================================================
    with tab3:
        st.header("Resultados del Análisis")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Ejecuta el análisis primero")
            return
        
        results = st.session_state['results']
        analyzer = st.session_state['analyzer']
        
        # Métricas
        st.subheader("📊 Resumen")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔴 Vacancias", results['n_vacancies'],
                     delta=f"{results['vacancy_concentration']*100:.4f}%")
        
        with col2:
            st.metric("🟠 Intersticiales", results['n_interstitials'],
                     delta=f"{results['interstitial_concentration']*100:.4f}%")
        
        with col3:
            st.metric("⚖️ Balance", f"{analyzer.n_atoms - analyzer.n_sites:+d}",
                     delta="átomos - sitios")
        
        with col4:
            st.metric("📐 Strain", f"{results['volumetric_strain']*100:.2f}%",
                     delta="volumétrico")
        
        st.markdown("---")
        
        # Visualización principal
        st.subheader("🎨 Visualización 3D Interactiva")
        st.markdown("---")
        st.subheader("💾 Exportar Defectos")

        if st.button("Exportar archivo .dump de defectos"):
            output_path = os.path.join(tempfile.gettempdir(), "defects_only.dump")
            filename = export_defects_dump(analyzer, results, output_path)
            with open(filename, "rb") as f:
                st.download_button(
                    label="⬇️ Descargar defects_only.dump",
                    data=f,
                    file_name="defects_only.dump",
                    mime="text/plain"
                )
                
        col1, col2 = st.columns([3, 1])
        
        with col2:
            use_auto_range = st.checkbox(
                "🎯 Ajuste automático de rango",
                value=True,
                help="Ajusta el rango automáticamente según las dimensiones de la celda de simulación"
            )
            
            if not use_auto_range:
                plot_range = st.slider(
                    "Rango manual (Å)", 5.0, 100.0, 20.0, 5.0,
                    help="Tamaño de la región visible (desde -rango hasta +rango)"
                )
            else:
                plot_range = 20.0  # Valor por defecto, no se usa
                ref_box = st.session_state.get('ref_box', None)
                if ref_box is not None:
                    st.info(f"""
                    📏 **Dimensiones de la celda:**
                    - X: {ref_box.lx:.2f} Å ({ref_box.xlo:.2f} → {ref_box.xhi:.2f})
                    - Y: {ref_box.ly:.2f} Å ({ref_box.ylo:.2f} → {ref_box.yhi:.2f})
                    - Z: {ref_box.lz:.2f} Å ({ref_box.zlo:.2f} → {ref_box.zhi:.2f})
                    """)
            
            st.markdown("---")
            
            show_normal = st.checkbox("Átomos normales", value=True)
            show_vacancies = st.checkbox("Vacancias", value=True)
            show_interstitials = st.checkbox("Intersticiales", value=True)
            
            st.markdown("---")
            
            st.info("💡 **Controles:**\n- Arrastra para rotar\n- Scroll para zoom\n- Hover para info")
        
        with col1:
            ref_box = st.session_state.get('ref_box', None)
            fig = create_3d_scatter_interactive(
                analyzer, results, plot_range,
                show_normal, show_vacancies, show_interstitials,
                use_auto_range=use_auto_range,
                ref_box=ref_box
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detalles
        st.subheader("📋 Detalles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Distribución de ocupación:**")
            occupancy_counts = np.bincount(analyzer.occupancy)
            
            for occ, count in enumerate(occupancy_counts):
                if count > 0:
                    if occ == 0:
                        st.write(f"- Ocupación {occ} (vacancia): {count} sitios")
                    elif occ == 1:
                        st.write(f"- Ocupación {occ} (normal): {count} sitios")
                    else:
                        st.write(f"- Ocupación {occ} (intersticiales): {count} sitios")
        
        with col2:
            st.write("**Parámetros:**")
            st.write(f"- Sitios de referencia: {analyzer.n_sites}")
            st.write(f"- Átomos defectuosos: {analyzer.n_atoms}")
            st.write(f"- PBC: {'Sí' if analyzer.use_pbc else 'No'}")
            st.write(f"- Mapeo afín: {'Sí' if analyzer.use_affine_mapping else 'No'}")
    
    # ========================================================================
    # TAB 4: VISUALIZACIONES AVANZADAS
    # ========================================================================
    with tab4:
        st.header("Visualizaciones Avanzadas")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Ejecuta el análisis primero")
            return
        
        results = st.session_state['results']
        analyzer = st.session_state['analyzer']
        
        st.subheader("🎨 Selecciona visualización")
        
        viz_type = st.selectbox(
            "Tipo",
            [
                "1️⃣ Comparación Lado a Lado",
                "2️⃣ Planos de Corte 2D",
                "3️⃣ Histogramas de Densidad 3D",
                "4️⃣ Análisis de Vecinos Cercanos"
            ]
        )
        
        # Parámetros
        st.sidebar.subheader("⚙️ Parámetros")
        plot_range_adv = st.sidebar.slider(
            "Rango (Å)", 5.0, 50.0, 25.0, 5.0, key="adv_range"
        )
        
        st.markdown("---")
        
        # Generar visualización
        if viz_type == "1️⃣ Comparación Lado a Lado":
            st.info("""
            **Izquierda:** Referencia con vacancias en rojo
            **Derecha:** Defectuosa con intersticiales en naranja
            
            💡 Rota ambos gráficos de forma independiente
            """)
            
            fig = create_comparison_interactive(analyzer, results, plot_range_adv)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "2️⃣ Planos de Corte 2D":
            slice_thickness = st.sidebar.slider(
                "Espesor (Å)", 2.0, 10.0, 5.0, 1.0, key="slice"
            )
            
            st.info(f"""
            Tres planos ortogonales (XY, XZ, YZ) centrados en el origen
            
            **Espesor:** ±{slice_thickness} Å del plano central
            """)
            
            fig = create_slices_interactive(analyzer, results, plot_range_adv, slice_thickness)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "3️⃣ Histogramas de Densidad 3D":
            st.info("""
            Distribución espacial de defectos en bins 3D
            
            💡 El tamaño y color muestran la concentración local
            """)
            
            fig = create_density_interactive(analyzer, results, plot_range_adv)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "4️⃣ Análisis de Vecinos Cercanos":
            st.info("""
            Histogramas de distancias al vecino más cercano
            
            **Ayuda a identificar:** Clustering o distribución uniforme
            """)
            
            fig = create_nearest_neighbor_plot(analyzer, results, plot_range_adv)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()