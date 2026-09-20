import os
import re
import io
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import t

import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# Configuración de estilos LaTeX para gráficas
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

st.set_page_config(
    page_title="Análisis Multifichero de Periodos",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Análisis Multifichero de Periodos (PDM, ANOVA, CLEANEST, FALC y Ajuste No Lineal)")

# ==========================================
# FUNCIONES MATEMÁTICAS Y DE ANÁLISIS
# ==========================================
def parse_uploaded_file(uploaded_file, skip_lines):
    parsed_rows = []
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("latin-1"))
        lines = stringio.readlines()
        
        data_lines = lines[skip_lines:]
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line and not re.search(r'\d+,\d+', line):
                parts = [p.strip() for p in line.split(',') if p.strip()]
            else:
                line_clean = line.replace(',', '.')
                parts = re.split(r'[\s\t]+', line_clean)
            if len(parts) >= 2:
                parsed_rows.append(parts)
                
        if not parsed_rows:
            return None, None, None
            
        df = pd.DataFrame(parsed_rows)
        
        def to_clean_float(series):
            cleaned = series.astype(str).str.replace(',', '.', regex=False).str.strip()
            return pd.to_numeric(cleaned, errors='coerce')
        
        times = to_clean_float(df.iloc[:, 0]).values
        magnitudes = to_clean_float(df.iloc[:, 1]).values
        
        valid_idx = ~np.isnan(times) & ~np.isnan(magnitudes)
        times = times[valid_idx]
        magnitudes = magnitudes[valid_idx]
        
        errors = None
        if df.shape[1] >= 3:
            raw_errs = to_clean_float(df.iloc[:, 2]).values
            raw_errs = raw_errs[valid_idx]
            if not np.all(np.isnan(raw_errs)):
                errors = np.where(raw_errs <= 0, 0.01, raw_errs)
                
        return times, magnitudes, errors
    except Exception as e:
        st.error(f"Error al leer el archivo {uploaded_file.name}: {e}")
        return None, None, None

def compute_pdm_theta(times, magnitudes, trial_period, num_bins):
    phases = (times / trial_period) % 1.0
    total_variance = np.var(magnitudes, ddof=1)
    if total_variance == 0:
        return 1.0
    
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    sum_bin_variances, total_points_in_bins = 0, 0
    
    for i in range(num_bins):
        idx = (phases >= bin_edges[i]) & (phases < bin_edges[i+1])
        bin_data = magnitudes[idx]
        n_i = len(bin_data)
        if n_i > 1:
            sum_bin_variances += (n_i - 1) * np.var(bin_data, ddof=1)
            total_points_in_bins += (n_i - 1)

    if total_points_in_bins == 0:
        return 1.0
    return (sum_bin_variances / total_points_in_bins) / total_variance

def compute_anova_f(times, magnitudes, trial_period, num_bins):
    phases = (times / trial_period) % 1.0
    n_total = len(magnitudes)
    global_mean = np.mean(magnitudes)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    
    active_bin_means, active_bin_counts = [], []
    ss_within = 0.0  
    
    for i in range(num_bins):
        idx = (phases >= bin_edges[i]) & (phases < bin_edges[i+1])
        bin_data = magnitudes[idx]
        n_i = len(bin_data)
        if n_i > 0:
            mean_i = np.mean(bin_data)
            active_bin_means.append(mean_i)
            active_bin_counts.append(n_i)
            ss_within += np.sum((bin_data - mean_i)**2)
    
    k_active = len(active_bin_means)
    if k_active <= 1 or (n_total - k_active) <= 0:
        return 0.0
        
    ss_between = sum([active_bin_counts[i] * (active_bin_means[i] - global_mean)**2 for i in range(k_active)])
    ms_between = ss_between / (k_active - 1)
    ms_within = ss_within / (n_total - k_active)
    
    return (ms_between / ms_within) if ms_within > 0 else 0.0

def compute_cleanest_power(times, magnitudes, trial_period):
    omega = 2.0 * np.pi / trial_period
    cos_w, sin_w = np.cos(omega * times), np.sin(omega * times)
    
    C, S, V = cos_w - np.mean(cos_w), sin_w - np.mean(sin_w), magnitudes - np.mean(magnitudes)
    cc, ss, cs = np.sum(C * C), np.sum(S * S), np.sum(C * S)
    vc, vs = np.sum(V * C), np.sum(V * S)
    
    determinant = cc * ss - cs * cs
    if abs(determinant) < 1e-10:
        return 0.0
        
    a = (vc * ss - vs * cs) / determinant
    b = (vs * cc - vc * cs) / determinant
    
    ss_total = np.sum(V * V)
    return ((a * vc + b * vs) / ss_total) if ss_total > 0 else 0.0

def compute_falc_full(times, magnitudes, errors, file_indices, trial_freq, harmonics):
    num_points = len(times)
    unique_files = np.unique(file_indices)
    num_files = len(unique_files)
    
    num_params = num_files + 2 * harmonics
    M = np.zeros((num_points, num_params))
    
    for i, file_id in enumerate(unique_files):
        M[file_indices == file_id, i] = 1.0
        
    omega = 2.0 * np.pi * trial_freq
    for k in range(1, harmonics + 1):
        col_sin = num_files + 2 * (k - 1)
        col_cos = col_sin + 1
        M[:, col_sin] = np.sin(k * omega * times)
        M[:, col_cos] = np.cos(k * omega * times)
        
    err_vec = errors if (errors is not None and len(errors) == num_points and np.any(errors > 0)) else np.full(num_points, 0.01)
    weights = 1.0 / (err_vec ** 2)
    W = np.diag(weights)
    
    MT_W = M.T @ W
    A_matrix = MT_W @ M
    B_vector = MT_W @ magnitudes
    
    try:
        params = np.linalg.solve(A_matrix, B_vector)
    except np.linalg.LinAlgError:
        return 1e9, None

    fitted_mags = M @ params
    residuals = magnitudes - fitted_mags
    chi_sq = np.sum(weights * (residuals ** 2))
    
    zero_points = {unique_files[i]: params[i] for i in range(num_files)}
    return chi_sq, zero_points

def fit_matlab_nlinfit(times, magnitudes, file_indices, initial_period, harmonics):
    unique_files = np.unique(file_indices)
    num_files = len(unique_files)
    
    t0 = times[0]
    t_rel = 24.0 * (times - t0)
    w_init = 2.0 * np.pi / (initial_period * 24.0)

    p0 = [np.mean(magnitudes)] + [0.0] * (2 * harmonics) + [w_init]
    if num_files > 1:
        p0 += [0.0] * (num_files - 1)

    def fourier_nlin_model(t, *params):
        a0 = params[0]
        w = params[2 * harmonics + 1]
        m = a0 * np.ones_like(t)
        
        for k in range(1, harmonics + 1):
            ak = params[2 * k - 1]
            bk = params[2 * k]
            m += ak * np.cos(k * w * t) + bk * np.sin(k * w * t)
            
        if num_files > 1:
            offsets = [0.0] + list(params[2 * harmonics + 2:])
            for idx, f_id in enumerate(unique_files):
                m[file_indices == f_id] += offsets[idx]
        return m

    try:
        popt, pcov = curve_fit(fourier_nlin_model, t_rel, magnitudes, p0=p0, maxfev=10000)
        w_fit = popt[2 * harmonics + 1]
        period_fit_hours = 2.0 * np.pi / w_fit
        period_fit_days = period_fit_hours / 24.0
        
        perr = np.sqrt(np.diag(pcov))
        w_err = perr[2 * harmonics + 1]
        dof = max(1, len(times) - len(popt))
        t_val = t.ppf(0.975, dof)
        
        w_min = w_fit - t_val * w_err
        w_max = w_fit + t_val * w_err
        
        p_ci_lower = (2.0 * np.pi / w_max) / 24.0
        p_ci_upper = (2.0 * np.pi / w_min) / 24.0
        ci_str = f"IC 95%: [{p_ci_lower:.6f}, {p_ci_upper:.6f}] d"
        
        zp_dict = {}
        if num_files > 1:
            offsets = [0.0] + list(popt[2 * harmonics + 2:])
            for idx, f_id in enumerate(unique_files):
                zp_dict[f_id] = -offsets[idx]
        else:
        
            zp_dict[unique_files[0]] = 0.0

        return period_fit_days, ci_str, zp_dict
    except Exception:
        return initial_period, "IC 95%: Error en ajuste", {}

def fit_fourier_model(phases_norm, magnitudes, harmonics):
    n_pts = len(phases_norm)
    if n_pts < (2 * harmonics + 1) or harmonics <= 0:
        mean_val = np.mean(magnitudes) if n_pts > 0 else 0
        return mean_val * np.ones_like(magnitudes), mean_val * np.ones(400), np.linspace(-0.5, 1.5, 400)

    A = [np.ones_like(phases_norm)]
    for k in range(1, harmonics + 1):
        A.append(np.cos(2.0 * np.pi * k * phases_norm))
        A.append(np.sin(2.0 * np.pi * k * phases_norm))
    A = np.column_stack(A)

    coeffs, _, _, _ = np.linalg.lstsq(A, magnitudes, rcond=None)
    fitted_data = A @ coeffs

    grid_phases = np.linspace(-0.5, 1.5, 400)
    grid_norm = grid_phases % 1.0
    A_grid = [np.ones_like(grid_norm)]
    for k in range(1, harmonics + 1):
        A_grid.append(np.cos(2.0 * np.pi * k * grid_norm))
        A_grid.append(np.sin(2.0 * np.pi * k * grid_norm))
    A_grid = np.column_stack(A_grid)
    grid_fitted = A_grid @ coeffs

    return fitted_data, grid_fitted, grid_phases

# ==========================================
# INTERFAZ EN STREAMLIT (SIDEBAR Y CONTROLES)
# ==========================================
with st.sidebar:
    st.header("📂 Gestión de Archivos")
    star_name = st.text_input("Nombre del Objeto / Asteroide", value="")
    uploaded_files = st.file_uploader("Sube ficheros de texto (.txt / .csv)", type=["txt", "dat", "csv"], accept_multiple_files=True)
    
    skip_lines = st.number_input("Líneas de encabezado a descartar", min_value=0, value=0, step=1)

    st.markdown("---")
    st.header("⚙️ Configuración del Análisis")
    method = st.selectbox("Método de Análisis", ["CLEANEST (DCDFT)", "ANOVA (AoV)", "PDM (Stellingwerf)", "FALC (Fourier)", "Ajuste No-Lineal (MATLAB)"])
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p_min = st.number_input("P. Mínimo (d)", value=0.1, format="%.4f")
        p_step = st.number_input("Paso de Periodo", value=0.001, format="%.5f")
    with col_p2:
        p_max = st.number_input("P. Máximo (d)", value=5.0, format="%.4f")
        
    bins_val = st.number_input("Bins", min_value=2, value=10, step=1)
    harmonics_val = st.number_input("Armónicos (Fourier)", min_value=1, value=3, step=1)

    st.markdown("---")
    st.header("🔍 Filtrado y Correcciones")
    use_sigma_clip = st.checkbox("Filtrar Outliers (> N·σ)")
    sigma_val = st.number_input("Valor N (Sigma)", value=2.0, format="%.1f")
    
    run_btn = st.button("🚀 EJECUTAR ANÁLISIS", type="primary", use_container_width=True)

# Inicializar Session State para almacenar datos combinados y resultados
if "combined_times" not in st.session_state:
    st.session_state["combined_times"] = None
    st.session_state["combined_mags"] = None
    st.session_state["combined_errs"] = None
    st.session_state["combined_file_idx"] = None
    st.session_state["file_names"] = []
    st.session_state["y_offsets"] = {}
    st.session_state["trial_periods"] = None
    st.session_state["statistic_values"] = None
    st.session_state["current_period"] = None
    st.session_state["phase_offset"] = 0.0
    st.session_state["ci_95_str"] = ""
    st.session_state["has_errors"] = False

if run_btn and uploaded_files:
    all_times, all_mags, all_errs, all_file_idx = [], [], [], []
    file_names = []
    has_any_errors = False
    
    for i, file_obj in enumerate(uploaded_files):
        t_vals, m_vals, e_vals = parse_uploaded_file(file_obj, skip_lines)
        if t_vals is None or len(t_vals) == 0:
            continue
            
        all_times.append(t_vals)
        all_mags.append(m_vals)
        all_file_idx.append(np.full(len(t_vals), i))
        file_names.append(file_obj.name)
        
        if e_vals is not None:
            all_errs.append(e_vals)
            has_any_errors = True
        else:
            all_errs.append(np.full(len(t_vals), 0.01))
            
    if all_times:
        st.session_state["combined_times"] = np.concatenate(all_times)
        st.session_state["combined_mags"] = np.concatenate(all_mags)
        st.session_state["combined_errs"] = np.concatenate(all_errs)
        st.session_state["combined_file_idx"] = np.concatenate(all_file_idx)
        st.session_state["file_names"] = file_names
        st.session_state["has_errors"] = has_any_errors
        st.session_state["y_offsets"] = {i: 0.0 for i in range(len(file_names))}

        # Generar malla de periodos
        if "FALC" in method:
            f_min = 1.0 / p_max
            f_max = 1.0 / p_min
            trial_freqs = np.arange(f_min, f_max + p_step, p_step)
            trial_periods = 1.0 / trial_freqs
        else:
            trial_periods = np.arange(p_min, p_max, p_step)
            
        st.session_state["trial_periods"] = trial_periods
        
        mags_for_calc = np.copy(st.session_state["combined_mags"])
        stat_list = []
        
        for p_val in trial_periods:
            if "CLEANEST" in method:
                val = compute_cleanest_power(st.session_state["combined_times"], mags_for_calc, p_val)
            elif "ANOVA" in method:
                val = compute_anova_f(st.session_state["combined_times"], mags_for_calc, p_val, bins_val)
            elif "PDM" in method:
                val = compute_pdm_theta(st.session_state["combined_times"], mags_for_calc, p_val, bins_val)
            elif "FALC" in method or "Ajuste No-Lineal" in method:
                val, _ = compute_falc_full(st.session_state["combined_times"], mags_for_calc, st.session_state["combined_errs"], st.session_state["combined_file_idx"], 1.0 / p_val, harmonics_val)
            stat_list.append(val)
            
        statistic_values = np.nan_to_num(np.array(stat_list))
        st.session_state["statistic_values"] = statistic_values
        
        if "PDM" in method or "FALC" in method or "Ajuste No-Lineal" in method:
            best_idx = np.argmin(statistic_values)
        else:
            best_idx = np.argmax(statistic_values)
            
        st.session_state["current_period"] = trial_periods[best_idx]
        st.session_state["phase_offset"] = 0.0
        st.session_state["ci_95_str"] = ""

        if "FALC" in method:
            _, best_zp = compute_falc_full(st.session_state["combined_times"], st.session_state["combined_mags"], st.session_state["combined_errs"], st.session_state["combined_file_idx"], 1.0 / st.session_state["current_period"], harmonics_val)
            if best_zp:
                global_mean_zp = np.mean(list(best_zp.values()))
                for f_idx, c_m in best_zp.items():
                    st.session_state["y_offsets"][f_idx] = -(c_m - global_mean_zp)
        elif "Ajuste No-Lineal" in method:
            best_p, ci_txt, zp_fit = fit_matlab_nlinfit(
                st.session_state["combined_times"], st.session_state["combined_mags"], st.session_state["combined_file_idx"], st.session_state["current_period"], harmonics_val
            )
            st.session_state["current_period"] = best_p
            st.session_state["ci_95_str"] = ci_txt
            if zp_fit:
                for f_idx, off_val in zp_fit.items():
                    st.session_state["y_offsets"][f_idx] = off_val
                    
        st.success("¡Análisis ejecutado con éxito!")

# ==========================================
# VISUALIZACIÓN DE RESULTADOS Y GRÁFICAS
# ==========================================
if st.session_state["current_period"] is not None:
    st.markdown("---")
    col_res1, col_res2 = st.columns([3, 1])
    with col_res1:
        star_str = f"**Objeto:** {star_name} | " if star_name else ""
        st.markdown(f"### {star_str}Periodo Óptimo: `{st.session_state['current_period']:.6f} d` {st.session_state['ci_95_str']}")
    with col_res2:
        if st.button("Ajuste Automático Vertical (Zero-Point)"):
            # Lógica de alineación vertical automática
            phases = (st.session_state["combined_times"] / st.session_state["current_period"]) % 1.0
            num_phase_bins = 20
            bin_edges = np.linspace(0.0, 1.0, num_phase_bins + 1)
            bin_means = np.zeros(num_phase_bins)
            for b in range(num_phase_bins):
                idx = (phases >= bin_edges[b]) & (phases < bin_edges[b+1])
                bin_means[b] = np.mean(st.session_state["combined_mags"][idx]) if np.sum(idx) > 0 else np.nan

            for f_idx in np.unique(st.session_state["combined_file_idx"]):
                mask = (st.session_state["combined_file_idx"] == f_idx)
                f_phases = phases[mask]
                f_mags = st.session_state["combined_mags"][mask]
                diffs = []
                for j, ph in enumerate(f_phases):
                    b = int(ph * num_phase_bins)
                    if b >= num_phase_bins: b = num_phase_bins - 1
                    if not np.isnan(bin_means[b]):
                        diffs.append(bin_means[b] - f_mags[j])
                st.session_state["y_offsets"][f_idx] = np.mean(diffs) if diffs else 0.0
            st.rerun()

    # Controles manuales de offsets individuales por fichero
    if len(st.session_state["file_names"]) > 1:
        with st.expander("🛠️ Ajuste Manual de Offsets Verticales por Fichero"):
            cols = st.columns(len(st.session_state["file_names"]))
            for idx, fname in enumerate(st.session_state["file_names"]):
                with cols[idx]:
                    curr_off = st.session_state["y_offsets"].get(idx, 0.0)
                    new_off = st.number_input(f"{fname}", value=curr_off, format="%.3f", key=f"offset_{idx}")
                    st.session_state["y_offsets"][idx] = new_off

    # Dibujado del panel triple con Matplotlib
    fig = Figure(figsize=(12, 7), dpi=100)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1.0], height_ratios=[1.0, 0.55], hspace=0.2, wspace=0.22)
    ax1 = fig.add_subplot(gs[:, 0])             
    ax2 = fig.add_subplot(gs[0, 1])             
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2) 

    # 1. Periodograma
    plot_color = '#d32f2f' if ("FALC" in method or "Ajuste" in method) else ('#9c27b0' if "CLEANEST" in method else ('#1f77b4' if "ANOVA" in method else '#2ca02c'))
    ylabel_str = r'$\chi^2$' if ("FALC" in method or "Ajuste" in method) else ('Potencia' if "CLEANEST" in method else 'Estadístico')

    ax1.plot(st.session_state["trial_periods"], st.session_state["statistic_values"], color=plot_color, lw=1.2)
    ax1.axvline(st.session_state["current_period"], color='red', linestyle='--', alpha=0.9, label=f'P = {st.session_state["current_period"]:.5f} d')
    ax1.set_xlabel('Periodo de prueba P (días)')
    ax1.set_ylabel(ylabel_str)
    ax1.set_title(f"Periodograma - {star_name}" if star_name else "Periodograma")
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # Preparar datos plegados
    base_phases = (st.session_state["combined_times"] / st.session_state["current_period"]) % 1.0
    mags_with_offset = np.copy(st.session_state["combined_mags"])
    for f_idx, offset in st.session_state["y_offsets"].items():
        mags_with_offset[st.session_state["combined_file_idx"] == f_idx] += offset

    # Sigma clipping mask
    valid_points_mask = np.ones(len(mags_with_offset), dtype=bool)
    if use_sigma_clip:
        num_b = 25
        b_edges = np.linspace(0.0, 1.0, num_b + 1)
        for b in range(num_b):
            idx_b = (base_phases >= b_edges[b]) & (base_phases < b_edges[b+1])
            if np.sum(idx_b) > 2:
                b_data = mags_with_offset[idx_b]
                mean_b, std_b = np.mean(b_data), np.std(b_data, ddof=1)
                if std_b > 0:
                    out = np.abs(b_data - mean_b) > (sigma_val * std_b)
                    valid_points_mask[np.where(idx_b)[0][out]] = False

    adjusted_phases = base_phases - st.session_state["phase_offset"]
    norm_valid_phases = (adjusted_phases[valid_points_mask]) % 1.0
    fitted_data, grid_fitted, grid_phases = fit_fourier_model(norm_valid_phases, mags_with_offset[valid_points_mask], harmonics_val)
    residuals_valid = mags_with_offset[valid_points_mask] - fitted_data
    rms_residuals = np.sqrt(np.mean(residuals_valid**2)) if len(residuals_valid) > 0 else 0.0

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # 2. Curva Plegada
    for file_i in np.unique(st.session_state["combined_file_idx"]):
        color = color_palette[int(file_i) % len(color_palette)]
        file_name = st.session_state["file_names"][int(file_i)]
        offset_y = st.session_state["y_offsets"].get(file_i, 0.0)
        
        mask = (st.session_state["combined_file_idx"] == file_i) & valid_points_mask
        sub_phases = adjusted_phases[mask]
        sub_mags = mags_with_offset[mask]
        
        phases_in_range, mags_in_range = [], []
        for j, ph in enumerate(sub_phases):
            norm_ph = ph % 1.0
            for shift in [-1, 0, 1]:
                if -0.5 <= norm_ph + shift <= 1.5:
                    phases_in_range.append(norm_ph + shift)
                    mags_in_range.append(sub_mags[j])

        label_text = f"{file_name} (Δy={offset_y:+.3f})" if offset_y != 0 else file_name
        ax2.scatter(phases_in_range, mags_in_range, color=color, s=12, alpha=0.7, label=label_text)

    ax2.plot(grid_phases, grid_fitted, color='#7f7f7f', linestyle='--', lw=1.5, label=f'Fourier (K={harmonics_val})')
    ax2.set_ylabel('Magnitud m')
    ax2.set_title(f"Curva Plegada (P = {st.session_state['current_period']:.5f} d)")
    ax2.set_xlim(-0.5, 1.5)
    ax2.grid(True, linestyle=':')
    ax2.tick_params(labelbottom=False)
    ax2.legend(fontsize=7, loc='best')
    ax2.invert_yaxis()

    # 3. Residuos
    for file_i in np.unique(st.session_state["combined_file_idx"]):
        color = color_palette[int(file_i) % len(color_palette)]
        mask_sub = (st.session_state["combined_file_idx"][valid_points_mask] == file_i)
        sub_phases = adjusted_phases[valid_points_mask][mask_sub]
        sub_res = residuals_valid[mask_sub]

        res_phases_in_range, res_in_range = [], []
        for j, ph in enumerate(sub_phases):
            norm_ph = ph % 1.0
            for shift in [-1, 0, 1]:
                if -0.5 <= norm_ph + shift <= 1.5:
                    res_phases_in_range.append(norm_ph + shift)
                    res_in_range.append(sub_res[j])

        ax3.scatter(res_phases_in_range, res_in_range, color=color, s=10, alpha=0.7)

    ax3.axhline(0.0, color='red', linestyle='--', alpha=0.8, lw=1.0)
    ax3.set_xlabel('Fase φ')
    ax3.set_ylabel('Residuo Δm')
    ax3.set_title(f"Residuos (RMS = {rms_residuals:.4f} mag)", fontsize=9)
    ax3.set_xlim(-0.5, 1.5)
    ax3.grid(True, linestyle=':')
    ax3.invert_yaxis()
    fig.tight_layout()

    st.pyplot(fig)

    # ==========================================
    # SECCIÓN DE EXPORTACIONES Y DESCARGAS
    # ==========================================
    st.markdown("---")
    st.subheader("💾 Exportación de Resultados y Modelos")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Exportar Fases Ordenadas
        base_ph = (st.session_state["combined_times"] / st.session_state["current_period"]) % 1.0
        val_ph = (base_ph[valid_points_mask] - st.session_state["phase_offset"]) % 1.0
        val_mg = mags_with_offset[valid_points_mask]
        sort_idx = np.argsort(val_ph)
        
        csv_fases = pd.DataFrame({"Fase": val_ph[sort_idx], "Magnitud": val_mg[sort_idx]}).to_csv(index=False, sep="\t")
        st.download_button("📥 Descargar Fases Ordenadas (.txt)", data=csv_fases, file_name="fases_ordenadas.txt", mime="text/plain")

    with col_exp2:
        # Exportar Modelo de Fourier
        csv_modelo = pd.DataFrame({"Fase": grid_phases % 1.0, "Magnitud_Modelo": grid_fitted}).to_csv(index=False, sep="\t")
        st.download_button("📥 Descargar Modelo Fourier (.txt)", data=csv_modelo, file_name="modelo_fourier.txt", mime="text/plain")

    with col_exp3:
        # Exportar Residuos
        csv_res = pd.DataFrame({"Fase": norm_valid_phases, "Residuo": residuals_valid}).to_csv(index=False, sep="\t")
        st.download_button("📥 Descargar Residuos (.txt)", data=csv_res, file_name="residuos.txt", mime="text/plain")

else:
    st.info("👆 Por favor, sube uno o más archivos de texto en la barra lateral y haz clic en **EJECUTAR ANÁLISIS**.")