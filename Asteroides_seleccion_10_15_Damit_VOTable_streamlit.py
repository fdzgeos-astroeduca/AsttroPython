# Programa de Buscador de Asteroides - Adaptado para Streamlit

import os
import json
import re
import gzip
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

import streamlit as st
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body_barycentric_posvel, GCRS, Angle, SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.io.votable import from_table, writeto

# Intentamos importar pypdf para leer el índice del MPB
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Configuración inicial de la página en Streamlit
st.set_page_config(
    page_title="Buscador de Asteroides Observables",
    page_icon="🌌",
    layout="wide"
)

# ==========================================
# RUTAS DE DATOS LOCALES Y CONFIGURACIÓN
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPCORB_PATH = os.path.join(BASE_DIR, "MPCORB.DAT.gz")
ASTORB_PATH = os.path.join(BASE_DIR, "ASTORB.DAT.gz")
JPL_ROTATION_PATH = os.path.join(BASE_DIR, "jpl_rotation_periods.csv")
MPB_PDF_PATH = os.path.join(BASE_DIR, "MPB_LightcurveIndex.pdf")
CONFIG_PATH = os.path.join(BASE_DIR, "config_observatorio.json")

def cargar_configuracion():
    """Carga las últimas coordenadas guardadas o usa valores por defecto."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"lat": 37.3891, "lon": -5.9845}

def guardar_configuracion(lat, lon):
    """Guarda las coordenadas actuales."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"lat": lat, "lon": lon}, f)
    except Exception:
        pass

# ==========================================
# FUNCIONES DE DESCARGA, LIMPIEZA Y CÁLCULO
# ==========================================
@st.cache_resource
def descargar_mpcorb():
    url = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
    if not os.path.exists(MPCORB_PATH):
        r = requests.get(url, stream=True)
        with open(MPCORB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def descargar_astorb():
    url = "https://ftp.lowell.edu/pub/elgb/astorb.dat.gz"
    if not os.path.exists(ASTORB_PATH):
        r = requests.get(url, stream=True)
        with open(ASTORB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def descargar_jpl_rotacion():
    if os.path.exists(JPL_ROTATION_PATH):
        df = pd.read_csv(JPL_ROTATION_PATH)
        if "pdes" in df.columns:
            return df
            
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    constraint = json.dumps({"AND": ["rot_per|DF"]}, separators=(",", ":"))
    params = {"fields": "pdes,rot_per", "sb-kind": "a", "sb-cdata": constraint, "limit": "100000"}
    r = requests.get(url, params=params, timeout=30)
    data = r.json().get("data", [])
    df = pd.DataFrame(data, columns=["pdes", "rot_per_h"])
    df["rot_per_h"] = pd.to_numeric(df["rot_per_h"], errors="coerce")
    df["pdes"] = df["pdes"].astype(str).str.strip()
    df.to_csv(JPL_ROTATION_PATH, index=False)
    return df

def _jpl_pdes_from_mpc(designation):
    text = str(designation or "").strip()
    numbered = re.match(r"^\((\d+)\)", text)
    if numbered:
        return numbered.group(1)
    if re.fullmatch(r"\d+", text):
        return text
    return re.sub(r"\s+", " ", text).upper()

@st.cache_data
def cargar_mpcorb():
    descargar_mpcorb()
    colspecs = [(0, 7), (8, 13), (14, 19), (20, 25), (26, 35), (37, 46), (48, 57), (59, 68), (70, 79), (80, 91), (92, 103), (166, 194)]
    names = ["designation_packed", "H", "G", "epoch_packed", "M0_deg", "argperi_deg", "node_deg", "incl_deg", "e", "n_deg_day", "a_au", "designation"]
    df = pd.read_fwf(MPCORB_PATH, colspecs=colspecs, names=names, compression="gzip", skiprows=40)
    for col in ["H", "G", "M0_deg", "argperi_deg", "node_deg", "incl_deg", "e", "n_deg_day", "a_au"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["designation"] = df["designation"].fillna(df["designation_packed"]).astype(str).str.strip()
    df.dropna(subset=["a_au", "e", "incl_deg"], inplace=True)
    df['clean_des'] = df['designation'].apply(_jpl_pdes_from_mpc)
    return df

@st.cache_data
def cargar_astorb():
    if not os.path.exists(ASTORB_PATH):
        descargar_astorb()
        
    rows = []
    with gzip.open(ASTORB_PATH, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            try:
                num_str = line[0:6].strip()
                name_str = line[7:25].strip()
                sub_bv = line[52:58].strip()
                bv = float(sub_bv) if sub_bv and sub_bv.lower() not in ('n.a.', '') else np.nan

                num = int(num_str) if num_str.isdigit() else None
                des = name_str if name_str else (str(num) if num is not None else "")
                clean_key = str(num) if num is not None else re.sub(r"\s+", " ", des).upper()
                rows.append({"clean_key": clean_key, "B_V": bv})
            except Exception:
                continue
    return pd.DataFrame(rows)

def packed_epoch_to_jd(series):
    s = series.astype(str).to_numpy(dtype="U5")
    c0 = np.array([x[0] if len(x) >= 1 else "?" for x in s], dtype="U1")
    yy = np.array([int(x[1:3]) if len(x) >= 3 and x[1:3].isdigit() else -999 for x in s], dtype=np.int32)
    cm = np.array([x[3] if len(x) >= 4 else "?" for x in s], dtype="U1")
    cd = np.array([x[4] if len(x) >= 5 else "?" for x in s], dtype="U1")
    def char_val(chars):
        out = np.full(chars.shape, -1, dtype=np.int16)
        codes = np.char.upper(chars.astype("U1")).view("U1")
        for d in range(10): out[codes == str(d)] = d
        for j, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=10): out[codes == c] = j
        return out
    century = char_val(c0)
    year = century * 100 + yy
    month = char_val(cm)
    day = char_val(cd)
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn.astype(float) - 0.5

def calcular_posiciones(df, t_astropy, loc, return_eq=False):
    epoch_jd = packed_epoch_to_jd(df["epoch_packed"])
    a, e = df["a_au"].values, df["e"].values
    inc, node, argperi = np.deg2rad(df["incl_deg"].values), np.deg2rad(df["node_deg"].values), np.deg2rad(df["argperi_deg"].values)
    M0, n = np.deg2rad(df["M0_deg"].values), np.deg2rad(df["n_deg_day"].values)

    dt_days = t_astropy.tt.jd - epoch_jd
    M = np.mod(M0 + n * dt_days, 2.0 * np.pi)
    E = M + e * np.sin(M) * (1.0 + e * np.cos(M))
    for _ in range(5): E -= (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(np.maximum(0.0, 1.0 - e * e)) * np.sin(E)
    
    x_hel = (np.cos(node)*np.cos(argperi) - np.sin(node)*np.sin(argperi)*np.cos(inc))*x_orb + (-np.cos(node)*np.sin(argperi) - np.sin(node)*np.cos(argperi)*np.cos(inc))*y_orb
    y_hel = (np.sin(node)*np.cos(argperi) + np.cos(node)*np.sin(argperi)*np.cos(inc))*x_orb + (-np.sin(node)*np.sin(argperi) + np.cos(node)*np.cos(argperi)*np.cos(inc))*y_orb
    z_hel = (np.sin(argperi)*np.sin(inc))*x_orb + (np.cos(argperi)*np.sin(inc))*y_orb

    earth_bary, _ = get_body_barycentric_posvel("earth", t_astropy)
    sun_bary, _ = get_body_barycentric_posvel("sun", t_astropy)
    earth_eq = (earth_bary.xyz - sun_bary.xyz).to_value(u.au)
    eps = np.deg2rad(23.439291111)
    earth_ecl = np.array([earth_eq[0], earth_eq[1]*np.cos(eps) + earth_eq[2]*np.sin(eps), -earth_eq[1]*np.sin(eps) + earth_eq[2]*np.cos(eps)])
    
    gx, gy, gz = x_hel - earth_ecl[0], y_hel - earth_ecl[1], z_hel - earth_ecl[2]
    delta, r = np.sqrt(gx**2 + gy**2 + gz**2), np.sqrt(x_hel**2 + y_hel**2 + z_hel**2)
    
    earth_sun_au = np.linalg.norm(earth_ecl, axis=0)
    
    H, G_param = df["H"].values, df["G"].fillna(0.15).values
    cos_alpha = (r**2 + delta**2 - earth_sun_au**2) / (2.0 * r * delta)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    tan_half = np.tan(alpha / 2.0)
    phase = np.maximum((1.0 - G_param) * np.exp(-3.33 * np.power(np.maximum(tan_half, 0.0), 0.63)) + G_param * np.exp(-1.87 * np.power(np.maximum(tan_half, 0.0), 1.22)), 1e-12)
    V_est = H + 5.0 * np.log10(r * delta) - 2.5 * np.log10(phase)
    
    gy_eq = gy * np.cos(eps) - gz * np.sin(eps)
    gz_eq = gy * np.sin(eps) + gz * np.cos(eps)
    
    gcrs = GCRS(x=gx*u.au, y=gy_eq*u.au, z=gz_eq*u.au, representation_type="cartesian", obstime=t_astropy)
    altaz = gcrs.transform_to(AltAz(obstime=t_astropy, location=loc, pressure=0*u.hPa))
    
    if return_eq:
        ra = gcrs.spherical.lon.deg
        dec = gcrs.spherical.lat.deg
        return altaz.alt.deg, V_est, ra, dec
    return altaz.alt.deg, V_est

# ==========================================
# INTERFAZ DE STREAMLIT
# ==========================================
st.title("🔭 Buscador de Asteroides Observables")

config = cargar_configuracion()

with st.sidebar:
    st.header("Parámetros de Observación")
    lat = st.number_input("Latitud (°)", value=float(config.get("lat", 37.3891)), format="%.4f")
    lon = st.number_input("Longitud (°)", value=float(config.get("lon", -5.9845)), format="%.4f")
    
    guardar_configuracion(lat, lon)
    
    ahora_utc = datetime.utcnow()
    fecha_def = st.text_input("Fecha (YYYY-MM-DD)", value=ahora_utc.strftime("%Y-%m-%d"))
    hora_def = st.text_input("Hora Inicio (UT)", value=ahora_utc.strftime("%H:%M:%S"))
    
    alt_min = st.number_input("Altura Mínima (°)", value=30.0)
    mag_min = st.number_input("Mag. Mínima (Brillante)", value=8.0)
    mag_max = st.number_input("Mag. Máxima (Débil)", value=16.0)
    
    rot_min = st.number_input("Periodo Rot. Mín (h)", value=2.0)
    rot_max = st.number_input("Periodo Rot. Máx (h)", value=12.0)
    
    modo_calculo = st.radio("Modo de búsqueda:", ["Con periodo de rotación", "Sin periodo de rotación"])
    
    btn_ejecutar = st.button("🚀 Ejecutar Cálculo", type="primary")

# Contenedor principal de resultados guardado en el session_state
if "resultados_df" not in st.session_state:
    st.session_state["resultados_df"] = None

if btn_ejecutar:
    with st.spinner("Cargando catálogos y calculando posiciones... Esto puede tomar un momento."):
        try:
            mpc_df = cargar_mpcorb()
            jpl_df = descargar_jpl_rotacion()
            astorb_df = cargar_astorb()
            
            if modo_calculo == "Con periodo de rotación":
                df = pd.merge(mpc_df, jpl_df, left_on="clean_des", right_on="pdes", how="inner")
                df = df[(df['rot_per_h'] >= rot_min) & (df['rot_per_h'] <= rot_max)].copy()
            else:
                df = pd.merge(mpc_df, jpl_df, left_on="clean_des", right_on="pdes", how="left")
                df = df[df['rot_per_h'].isna()].copy()
                df["rot_per_h"] = np.nan
            
            if df.empty:
                st.warning("Ningún asteroide cumple los filtros de rotación seleccionados.")
                st.session_state["resultados_df"] = None
            else:
                df = pd.merge(df, astorb_df, left_on="clean_des", right_on="clean_key", how="left")
                
                t_inicio = Time(f"{fecha_def} {hora_def}")
                loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
                
                altitudes, magnitudes, ras, decs = calcular_posiciones(df, t_inicio, loc, return_eq=True)
                df["Alt_T0"], df["Mag_T0"], df["RA_deg"], df["DEC_deg"] = altitudes, magnitudes, ras, decs
                
                visibles = df[(df["Alt_T0"] >= alt_min) & (df["Mag_T0"] >= mag_min) & (df["Mag_T0"] <= mag_max)].copy()
                
                if visibles.empty:
                    st.warning("Ningún asteroide cumple los criterios de visibilidad inicial.")
                    st.session_state["resultados_df"] = None
                else:
                    dt_base = datetime.strptime(f"{fecha_def} {hora_def}", "%Y-%m-%d %H:%M:%S")
                    intervalos_minutos = 15
                    tiempos_noche = [dt_base + timedelta(minutes=i*intervalos_minutos) for i in range(48)]
                    times_astropy = Time(tiempos_noche)
                    
                    sun_altaz = get_sun(times_astropy).transform_to(AltAz(obstime=times_astropy, location=loc, pressure=0*u.hPa))
                    sun_alts = sun_altaz.alt.deg

                    horas_visibles_list = []
                    for idx, row in visibles.iterrows():
                        temp_df = pd.DataFrame([row] * len(times_astropy))
                        altos, _ = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
                        tramos_visibles = np.sum((altos >= alt_min) & (sun_alts <= -12.0))
                        horas_visibles_list.append((tramos_visibles * intervalos_minutos) / 60.0)
                    
                    visibles["Horas_Visibles"] = horas_visibles_list
                    visibles = visibles.sort_values(by=["Horas_Visibles", "Mag_T0"], ascending=[False, True])
                    
                    # Formatear coordenadas para mostrar
                    tabla_final_rows = []
                    for _, row in visibles.iterrows():
                        ar_angle = Angle(row['RA_deg'], u.deg)
                        dec_angle = Angle(row['DEC_deg'], u.deg)
                        
                        ra_str = ar_angle.to_string(unit=u.hour, sep=':', precision=2, pad=True)
                        dec_str = dec_angle.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
                        dec_str = dec_str.replace(':', '°', 1).replace(':', "'", 1) + '"'
                        ra_str = ra_str.replace(':', 'h', 1).replace(':', 'm', 1) + 's'
                        
                        tabla_final_rows.append({
                            "Asteroide": row["designation"],
                            "AR (J2000)": ra_str,
                            "Dec (J2000)": dec_str,
                            "Altura Inicial": f"{row['Alt_T0']:.1f}°",
                            "Mag V": f"{row['Mag_T0']:.2f}",
                            "Rotación (h)": f"{row['rot_per_h']:.2f}" if pd.notna(row['rot_per_h']) else "Desconocido",
                            "Horas Visibles": f"{row['Horas_Visibles']:.1f}",
                            "_row_data": row.to_dict()
                        })
                    
                    st.session_state["resultados_df"] = pd.DataFrame(tabla_final_rows)
                    st.success(f"¡Cálculo finalizado! Se encontraron {len(visibles)} asteroides.")
        except Exception as e:
            st.error(f"Ha ocurrido un error durante el cálculo:\n{str(e)}")

# Mostrar resultados si existen
if st.session_state["resultados_df"] is not None:
    df_res = st.session_state["resultados_df"]
    st.subheader("Listado de Asteroides Observables")
    
    # Mostrar tabla limpia sin la columna interna de datos
    st.dataframe(df_res.drop(columns=["_row_data"]), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Herramientas y Gráficas Detalladas por Asteroide")
    
    # Selector de asteroide para ver gráficos y opciones adicionales
    nombres_asteroides = df_res["Asteroide"].tolist()
    asteroide_seleccionado = st.selectbox("Selecciona un asteroide de la lista para ver análisis detallado:", nombres_asteroides)
    
    if asteroide_seleccionado:
        row_sel_data = df_res[df_res["Asteroide"] == asteroide_seleccionado]["_row_data"].values[0]
        pdes = row_sel_data.get("clean_des", row_sel_data["designation"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Ver Gráfica de Altitud (Noche)"):
                dt_base = datetime.strptime(f"{fecha_def} {hora_def}", "%Y-%m-%d %H:%M:%S")
                intervalos_minutos = 15
                tiempos = [dt_base + timedelta(minutes=i*intervalos_minutos) for i in range(49)]
                times_astropy = Time(tiempos)
                
                temp_df = pd.DataFrame([row_sel_data] * len(times_astropy))
                loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
                altitudes, _ = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
                
                sun_altaz = get_sun(times_astropy).transform_to(AltAz(obstime=times_astropy, location=loc, pressure=0*u.hPa))
                sun_altitudes = sun_altaz.alt.deg
                etiquetas_tiempo = [t.strftime("%H:%M") for t in tiempos]
                
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(etiquetas_tiempo, altitudes, marker='o', color='dodgerblue', linewidth=2, label='Altura asteroide')
                ax.plot(etiquetas_tiempo, sun_altitudes, marker='x', color='gold', linewidth=1.5, linestyle='-.', label='Altura Sol')
                ax.axhline(y=alt_min, color='crimson', linestyle='--', linewidth=1.5, label=f'Altura mínima ({alt_min}°)')
                ax.axhline(y=-12.0, color='darkorange', linestyle='-', linewidth=1.5, label='Crepúsculo náutico (Sol -12°)')
                ax.set_title(f"Evolución de Altura - {asteroide_seleccionado}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Hora UT")
                ax.set_ylabel("Altura (°)")
                ax.grid(True, linestyle=':', alpha=0.7)
                plt.xticks(rotation=45)
                ax.legend(loc='upper right')
                st.pyplot(fig)

        with col2:
            if st.button("📉 Ver Variación de Magnitud (30 días)"):
                fecha_base = datetime.strptime(fecha_def, "%Y-%m-%d").date()
                fechas = [fecha_base + timedelta(days=i) for i in range(30)]
                tiempos = [datetime.combine(f, datetime.strptime(hora_def, "%H:%M:%S").time()) for f in fechas]
                times_astropy = Time(tiempos)
                
                temp_df = pd.DataFrame([row_sel_data] * len(times_astropy))
                loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
                _, magnitudes = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
                
                etiquetas_fechas = [f.strftime("%d/%m") for f in fechas]
                
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(etiquetas_fechas, magnitudes, marker='s', color='forestgreen', linewidth=2, label='Magnitud V estimada')
                ax.invert_yaxis()
                ax.set_title(f"Variación de Magnitud (30 días) - {asteroide_seleccionado}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Magnitud V (más brillo ↑)")
                ax.grid(True, linestyle=':', alpha=0.7)
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)

        with col3:
            st.markdown("### Enlaces Externos")
            st.markdown(f"- [Ver modelo 3D en DAMIT](https://damit.cuni.cz/projects/damit/?q={pdes})", unsafe_allow_html=True)
            st.markdown(f"- [Curva / JPL SBDB](https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={pdes})", unsafe_allow_html=True)
            st.markdown(f"- [Aladin Lite](https://aladin.cds.unistra.fr/AladinLite/?target={row_sel_data['RA_deg']}+{row_sel_data['DEC_deg']})", unsafe_allow_html=True)
            st.markdown(f"- [Buscador ALCDEF](https://alcdef.org)", unsafe_allow_html=True)# Programa de Buscador de Asteroides - Adaptado para Streamlit

import os
import json
import re
import gzip
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

import streamlit as st
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun, get_body_barycentric_posvel, GCRS, Angle, SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.io.votable import from_table, writeto

# Intentamos importar pypdf para leer el índice del MPB
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Configuración inicial de la página en Streamlit
st.set_page_config(
    page_title="Buscador de Asteroides Observables",
    page_icon="🌌",
    layout="wide"
)

# ==========================================
# RUTAS DE DATOS LOCALES Y CONFIGURACIÓN
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPCORB_PATH = os.path.join(BASE_DIR, "MPCORB.DAT.gz")
ASTORB_PATH = os.path.join(BASE_DIR, "ASTORB.DAT.gz")
JPL_ROTATION_PATH = os.path.join(BASE_DIR, "jpl_rotation_periods.csv")
MPB_PDF_PATH = os.path.join(BASE_DIR, "MPB_LightcurveIndex.pdf")
CONFIG_PATH = os.path.join(BASE_DIR, "config_observatorio.json")

def cargar_configuracion():
    """Carga las últimas coordenadas guardadas o usa valores por defecto."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"lat": 37.3891, "lon": -5.9845}

def guardar_configuracion(lat, lon):
    """Guarda las coordenadas actuales."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"lat": lat, "lon": lon}, f)
    except Exception:
        pass

# ==========================================
# FUNCIONES DE DESCARGA, LIMPIEZA Y CÁLCULO
# ==========================================
@st.cache_resource
def descargar_mpcorb():
    url = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
    if not os.path.exists(MPCORB_PATH):
        r = requests.get(url, stream=True)
        with open(MPCORB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def descargar_astorb():
    url = "https://ftp.lowell.edu/pub/elgb/astorb.dat.gz"
    if not os.path.exists(ASTORB_PATH):
        r = requests.get(url, stream=True)
        with open(ASTORB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def descargar_jpl_rotacion():
    if os.path.exists(JPL_ROTATION_PATH):
        df = pd.read_csv(JPL_ROTATION_PATH)
        if "pdes" in df.columns:
            return df
            
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    constraint = json.dumps({"AND": ["rot_per|DF"]}, separators=(",", ":"))
    params = {"fields": "pdes,rot_per", "sb-kind": "a", "sb-cdata": constraint, "limit": "100000"}
    r = requests.get(url, params=params, timeout=30)
    data = r.json().get("data", [])
    df = pd.DataFrame(data, columns=["pdes", "rot_per_h"])
    df["rot_per_h"] = pd.to_numeric(df["rot_per_h"], errors="coerce")
    df["pdes"] = df["pdes"].astype(str).str.strip()
    df.to_csv(JPL_ROTATION_PATH, index=False)
    return df

def _jpl_pdes_from_mpc(designation):
    text = str(designation or "").strip()
    numbered = re.match(r"^\((\d+)\)", text)
    if numbered:
        return numbered.group(1)
    if re.fullmatch(r"\d+", text):
        return text
    return re.sub(r"\s+", " ", text).upper()

@st.cache_data
def cargar_mpcorb():
    descargar_mpcorb()
    colspecs = [(0, 7), (8, 13), (14, 19), (20, 25), (26, 35), (37, 46), (48, 57), (59, 68), (70, 79), (80, 91), (92, 103), (166, 194)]
    names = ["designation_packed", "H", "G", "epoch_packed", "M0_deg", "argperi_deg", "node_deg", "incl_deg", "e", "n_deg_day", "a_au", "designation"]
    df = pd.read_fwf(MPCORB_PATH, colspecs=colspecs, names=names, compression="gzip", skiprows=40)
    for col in ["H", "G", "M0_deg", "argperi_deg", "node_deg", "incl_deg", "e", "n_deg_day", "a_au"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["designation"] = df["designation"].fillna(df["designation_packed"]).astype(str).str.strip()
    df.dropna(subset=["a_au", "e", "incl_deg"], inplace=True)
    df['clean_des'] = df['designation'].apply(_jpl_pdes_from_mpc)
    return df

@st.cache_data
def cargar_astorb():
    if not os.path.exists(ASTORB_PATH):
        descargar_astorb()
        
    rows = []
    with gzip.open(ASTORB_PATH, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            try:
                num_str = line[0:6].strip()
                name_str = line[7:25].strip()
                sub_bv = line[52:58].strip()
                bv = float(sub_bv) if sub_bv and sub_bv.lower() not in ('n.a.', '') else np.nan

                num = int(num_str) if num_str.isdigit() else None
                des = name_str if name_str else (str(num) if num is not None else "")
                clean_key = str(num) if num is not None else re.sub(r"\s+", " ", des).upper()
                rows.append({"clean_key": clean_key, "B_V": bv})
            except Exception:
                continue
    return pd.DataFrame(rows)

def packed_epoch_to_jd(series):
    s = series.astype(str).to_numpy(dtype="U5")
    c0 = np.array([x[0] if len(x) >= 1 else "?" for x in s], dtype="U1")
    yy = np.array([int(x[1:3]) if len(x) >= 3 and x[1:3].isdigit() else -999 for x in s], dtype=np.int32)
    cm = np.array([x[3] if len(x) >= 4 else "?" for x in s], dtype="U1")
    cd = np.array([x[4] if len(x) >= 5 else "?" for x in s], dtype="U1")
    def char_val(chars):
        out = np.full(chars.shape, -1, dtype=np.int16)
        codes = np.char.upper(chars.astype("U1")).view("U1")
        for d in range(10): out[codes == str(d)] = d
        for j, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=10): out[codes == c] = j
        return out
    century = char_val(c0)
    year = century * 100 + yy
    month = char_val(cm)
    day = char_val(cd)
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn.astype(float) - 0.5

def calcular_posiciones(df, t_astropy, loc, return_eq=False):
    epoch_jd = packed_epoch_to_jd(df["epoch_packed"])
    a, e = df["a_au"].values, df["e"].values
    inc, node, argperi = np.deg2rad(df["incl_deg"].values), np.deg2rad(df["node_deg"].values), np.deg2rad(df["argperi_deg"].values)
    M0, n = np.deg2rad(df["M0_deg"].values), np.deg2rad(df["n_deg_day"].values)

    dt_days = t_astropy.tt.jd - epoch_jd
    M = np.mod(M0 + n * dt_days, 2.0 * np.pi)
    E = M + e * np.sin(M) * (1.0 + e * np.cos(M))
    for _ in range(5): E -= (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(np.maximum(0.0, 1.0 - e * e)) * np.sin(E)
    
    x_hel = (np.cos(node)*np.cos(argperi) - np.sin(node)*np.sin(argperi)*np.cos(inc))*x_orb + (-np.cos(node)*np.sin(argperi) - np.sin(node)*np.cos(argperi)*np.cos(inc))*y_orb
    y_hel = (np.sin(node)*np.cos(argperi) + np.cos(node)*np.sin(argperi)*np.cos(inc))*x_orb + (-np.sin(node)*np.sin(argperi) + np.cos(node)*np.cos(argperi)*np.cos(inc))*y_orb
    z_hel = (np.sin(argperi)*np.sin(inc))*x_orb + (np.cos(argperi)*np.sin(inc))*y_orb

    earth_bary, _ = get_body_barycentric_posvel("earth", t_astropy)
    sun_bary, _ = get_body_barycentric_posvel("sun", t_astropy)
    earth_eq = (earth_bary.xyz - sun_bary.xyz).to_value(u.au)
    eps = np.deg2rad(23.439291111)
    earth_ecl = np.array([earth_eq[0], earth_eq[1]*np.cos(eps) + earth_eq[2]*np.sin(eps), -earth_eq[1]*np.sin(eps) + earth_eq[2]*np.cos(eps)])
    
    gx, gy, gz = x_hel - earth_ecl[0], y_hel - earth_ecl[1], z_hel - earth_ecl[2]
    delta, r = np.sqrt(gx**2 + gy**2 + gz**2), np.sqrt(x_hel**2 + y_hel**2 + z_hel**2)
    
    earth_sun_au = np.linalg.norm(earth_ecl, axis=0)
    
    H, G_param = df["H"].values, df["G"].fillna(0.15).values
    cos_alpha = (r**2 + delta**2 - earth_sun_au**2) / (2.0 * r * delta)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    tan_half = np.tan(alpha / 2.0)
    phase = np.maximum((1.0 - G_param) * np.exp(-3.33 * np.power(np.maximum(tan_half, 0.0), 0.63)) + G_param * np.exp(-1.87 * np.power(np.maximum(tan_half, 0.0), 1.22)), 1e-12)
    V_est = H + 5.0 * np.log10(r * delta) - 2.5 * np.log10(phase)
    
    gy_eq = gy * np.cos(eps) - gz * np.sin(eps)
    gz_eq = gy * np.sin(eps) + gz * np.cos(eps)
    
    gcrs = GCRS(x=gx*u.au, y=gy_eq*u.au, z=gz_eq*u.au, representation_type="cartesian", obstime=t_astropy)
    altaz = gcrs.transform_to(AltAz(obstime=t_astropy, location=loc, pressure=0*u.hPa))
    
    if return_eq:
        ra = gcrs.spherical.lon.deg
        dec = gcrs.spherical.lat.deg
        return altaz.alt.deg, V_est, ra, dec
    return altaz.alt.deg, V_est

# ==========================================
# INTERFAZ DE STREAMLIT
# ==========================================
st.title("🔭 Buscador de Asteroides Observables")

config = cargar_configuracion()

with st.sidebar:
    st.header("Parámetros de Observación")
    lat = st.number_input("Latitud (°)", value=float(config.get("lat", 37.3891)), format="%.4f")
    lon = st.number_input("Longitud (°)", value=float(config.get("lon", -5.9845)), format="%.4f")
    
    guardar_configuracion(lat, lon)
    
    ahora_utc = datetime.utcnow()
    fecha_def = st.text_input("Fecha (YYYY-MM-DD)", value=ahora_utc.strftime("%Y-%m-%d"))
    hora_def = st.text_input("Hora Inicio (UT)", value=ahora_utc.strftime("%H:%M:%S"))
    
    alt_min = st.number_input("Altura Mínima (°)", value=30.0)
    mag_min = st.number_input("Mag. Mínima (Brillante)", value=8.0)
    mag_max = st.number_input("Mag. Máxima (Débil)", value=16.0)
    
    rot_min = st.number_input("Periodo Rot. Mín (h)", value=2.0)
    rot_max = st.number_input("Periodo Rot. Máx (h)", value=12.0)
    
    modo_calculo = st.radio("Modo de búsqueda:", ["Con periodo de rotación", "Sin periodo de rotación"])
    
    btn_ejecutar = st.button("🚀 Ejecutar Cálculo", type="primary")

# Contenedor principal de resultados guardado en el session_state
if "resultados_df" not in st.session_state:
    st.session_state["resultados_df"] = None

if btn_ejecutar:
    with st.spinner("Cargando catálogos y calculando posiciones... Esto puede tomar un momento."):
        try:
            mpc_df = cargar_mpcorb()
            jpl_df = descargar_jpl_rotacion()
            astorb_df = cargar_astorb()
            
            if modo_calculo == "Con periodo de rotación":
                df = pd.merge(mpc_df, jpl_df, left_on="clean_des", right_on="pdes", how="inner")
                df = df[(df['rot_per_h'] >= rot_min) & (df['rot_per_h'] <= rot_max)].copy()
            else:
                df = pd.merge(mpc_df, jpl_df, left_on="clean_des", right_on="pdes", how="left")
                df = df[df['rot_per_h'].isna()].copy()
                df["rot_per_h"] = np.nan
            
            if df.empty:
                st.warning("Ningún asteroide cumple los filtros de rotación seleccionados.")
                st.session_state["resultados_df"] = None
            else:
                df = pd.merge(df, astorb_df, left_on="clean_des", right_on="clean_key", how="left")
                
                t_inicio = Time(f"{fecha_def} {hora_def}")
                loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
                
                altitudes, magnitudes, ras, decs = calcular_posiciones(df, t_inicio, loc, return_eq=True)
                df["Alt_T0"], df["Mag_T0"], df["RA_deg"], df["DEC_deg"] = altitudes, magnitudes, ras, decs
                
                visibles = df[(df["Alt_T0"] >= alt_min) & (df["Mag_T0"] >= mag_min) & (df["Mag_T0"] <= mag_max)].copy()
                
                if visibles.empty:
                    st.warning("Ningún asteroide cumple los criterios de visibilidad inicial.")
                    st.session_state["resultados_df"] = None
                else:
                    dt_base = datetime.strptime(f"{fecha_def} {hora_def}", "%Y-%m-%d %H:%M:%S")
                    intervalos_minutos = 15
                    tiempos_noche = [dt_base + timedelta(minutes=i*intervalos_minutos) for i in range(48)]
                    times_astropy = Time(tiempos_noche)
                    
                    sun_altaz = get_sun(times_astropy).transform_to(AltAz(obstime=times_astropy, location=loc, pressure=0*u.hPa))
                    sun_alts = sun_altaz.alt.deg

                    horas_visibles_list = []
                    for idx, row in visibles.iterrows():
                        temp_df = pd.DataFrame([row] * len(times_astropy))
                        altos, _ = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
                        tramos_visibles = np.sum((altos >= alt_min) & (sun_alts <= -12.0))
                        horas_visibles_list.append((tramos_visibles * intervalos_minutos) / 60.0)
                    
                    visibles["Horas_Visibles"] = horas_visibles_list
                    visibles = visibles.sort_values(by=["Horas_Visibles", "Mag_T0"], ascending=[False, True])
                    
                    # Formatear coordenadas para mostrar
                    tabla_final_rows = []
                    for _, row in visibles.iterrows():
                        ar_angle = Angle(row['RA_deg'], u.deg)
                        dec_angle = Angle(row['DEC_deg'], u.deg)
                        
                        ra_str = ar_angle.to_string(unit=u.hour, sep=':', precision=2, pad=True)
                        dec_str = dec_angle.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
                        dec_str = dec_str.replace(':', '°', 1).replace(':', "'", 1) + '"'
                        ra_str = ra_str.replace(':', 'h', 1).replace(':', 'm', 1) + 's'
                        
                        tabla_final_rows.append({
                            "Asteroide": row["designation"],
                            "AR (J2000)": ra_str,
                            "Dec (J2000)": dec_str,
                            "Altura Inicial": f"{row['Alt_T0']:.1f}°",
                            "Mag V": f"{row['Mag_T0']:.2f}",
                            "Rotación (h)": f"{row['rot_per_h']:.2f}" if pd.notna(row['rot_per_h']) else "Desconocido",
                            "Horas Visibles": f"{row['Horas_Visibles']:.1f}",
                            "_row_data": row.to_dict()
                        })
                    
                    st.session_state["resultados_df"] = pd.DataFrame(tabla_final_rows)
                    st.success(f"¡Cálculo finalizado! Se encontraron {len(visibles)} asteroides.")
        except Exception as e:
            st.error(f"Ha ocurrido un error durante el cálculo:\n{str(e)}")

# Mostrar resultados si existen
if st.session_state["resultados_df"] is not None:
    df_res = st.session_state["resultados_df"]
    st.subheader("Listado de Asteroides Observables")
    
    # Mostrar tabla limpia sin la columna interna de datos
    st.dataframe(df_res.drop(columns=["_row_data"]), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Herramientas y Gráficas Detalladas por Asteroide")
    
    # Selector de asteroide para ver gráficos y opciones adicionales
    nombres_asteroides = df_res["Asteroide"].tolist()
    asteroide_seleccionado = st.selectbox("Selecciona un asteroide de la lista para ver análisis detallado:", nombres_asteroides)
    
    if asteroide_seleccionado:
        row_sel_data = df_res[df_res["Asteroide"] == asteroide_seleccionado]["_row_data"].values[0]
        pdes = row_sel_data.get("clean_des", row_sel_data["designation"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Ver Gráfica de Altitud (Noche)"):
                dt_base = datetime.strptime(f"{fecha_def} {hora_def}", "%Y-%m-%d %H:%M:%S")
                intervalos_minutos = 15
                tiempos = [dt_base + timedelta(minutes=i*intervalos_minutos) for i in range(49)]
                times_astropy = Time(tiempos)
                
                temp_df = pd.DataFrame([row_sel_data] * len(times_astropy))
                loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
                altitudes, _ = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
                
                sun_altaz = get_sun(times_astropy).transform_to(AltAz(obstime=times_astropy, location=loc, pressure=0*u.hPa))
                sun_altitudes = sun_altaz.alt.deg
                etiquetas_tiempo = [t.strftime("%H:%M") for t in tiempos]
                
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(etiquetas_tiempo, altitudes, marker='o', color='dodgerblue', linewidth=2, label='Altura asteroide')
                ax.plot(etiquetas_tiempo, sun_altitudes, marker='x', color='gold', linewidth=1.5, linestyle='-.', label='Altura Sol')
                ax.axhline(y=alt_min, color='crimson', linestyle='--', linewidth=1.5, label=f'Altura mínima ({alt_min}°)')
                ax.axhline(y=-12.0, color='darkorange', linestyle='-', linewidth=1.5, label='Crepúsculo náutico (Sol -12°)')
                ax.set_title(f"Evolución de Altura - {asteroide_seleccionado}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Hora UT")
                ax.set_ylabel("Altura (°)")
                ax.grid(True, linestyle=':', alpha=0.7)
                plt.xticks(rotation=45)
                ax.legend(loc='upper right')
                st.pyplot(fig)

        with col2:
            if st.button("📉 Ver Variación de Magnitud (30 días)"):
                fecha_base = datetime.strptime(fecha_def, "%Y-%m-%d").date()
                fechas = [fecha_base + timedelta(days=i) for i in range(30)]
                tiempos = [datetime.combine(f, datetime.strptime(hora_def, "%H:%M:%S").time()) for f in fechas]
                times_astropy = Time(tiempos)
                
                temp_df = pd.DataFrame([row_sel_data] * len(times_astropy))
                loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
                _, magnitudes = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
                
                etiquetas_fechas = [f.strftime("%d/%m") for f in fechas]
                
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(etiquetas_fechas, magnitudes, marker='s', color='forestgreen', linewidth=2, label='Magnitud V estimada')
                ax.invert_yaxis()
                ax.set_title(f"Variación de Magnitud (30 días) - {asteroide_seleccionado}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Magnitud V (más brillo ↑)")
                ax.grid(True, linestyle=':', alpha=0.7)
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)

        with col3:
            st.markdown("### Enlaces Externos")
            st.markdown(f"- [Ver modelo 3D en DAMIT](https://damit.cuni.cz/projects/damit/?q={pdes})", unsafe_allow_html=True)
            st.markdown(f"- [Curva / JPL SBDB](https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={pdes})", unsafe_allow_html=True)
            st.markdown(f"- [Aladin Lite](https://aladin.cds.unistra.fr/AladinLite/?target={row_sel_data['RA_deg']}+{row_sel_data['DEC_deg']})", unsafe_allow_html=True)
            st.markdown(f"- [Buscador ALCDEF](https://alcdef.org)", unsafe_allow_html=True)