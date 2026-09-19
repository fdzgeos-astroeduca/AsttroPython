# Programa de Buscador de Asteroides - Con Gráfica de Altitud y Curva/Línea del Sol (-12°)

import os
import json
import re
import webbrowser
import gzip
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

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
    return {"lat": "37.3891", "lon": "-5.9845"}

def guardar_configuracion(lat, lon):
    """Guarda las coordenadas actuales antes de salir del programa."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"lat": lat, "lon": lon}, f)
    except Exception:
        pass

# ==========================================
# FUNCIONES DE DESCARGA, LIMPIEZA Y CÁLCULO
# ==========================================
def descargar_mpcorb():
    url = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
    if not os.path.exists(MPCORB_PATH):
        r = requests.get(url, stream=True)
        with open(MPCORB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

def descargar_astorb():
    url = "https://ftp.lowell.edu/pub/elgb/astorb.dat.gz"
    if not os.path.exists(ASTORB_PATH):
        r = requests.get(url, stream=True)
        with open(ASTORB_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

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

def cargar_mpcorb():
    colspecs = [(0, 7), (8, 13), (14, 19), (20, 25), (26, 35), (37, 46), (48, 57), (59, 68), (70, 79), (80, 91), (92, 103), (166, 194)]
    names = ["designation_packed", "H", "G", "epoch_packed", "M0_deg", "argperi_deg", "node_deg", "incl_deg", "e", "n_deg_day", "a_au", "designation"]
    df = pd.read_fwf(MPCORB_PATH, colspecs=colspecs, names=names, compression="gzip", skiprows=40)
    for col in ["H", "G", "M0_deg", "argperi_deg", "node_deg", "incl_deg", "e", "n_deg_day", "a_au"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["designation"] = df["designation"].fillna(df["designation_packed"]).astype(str).str.strip()
    df.dropna(subset=["a_au", "e", "incl_deg"], inplace=True)
    return df

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
# INTERFAZ GRÁFICA (GUI)
# ==========================================
class AppAsteroides(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Buscador de Asteroides Observables")
        self.geometry("1200x680")
        
        self.mpc_df = None
        self.jpl_df = None
        self.astorb_df = None
        self.asteroides_dict = {}
        
        # Cargar configuración previa (Latitud y Longitud)
        config = cargar_configuracion()
        lat_def = config.get("lat", "37.3891")
        lon_def = config.get("lon", "-5.9845")
        
        # Obtener fecha y hora actual del sistema (UTC para la hora de inicio UT)
        ahora_utc = datetime.utcnow()
        fecha_def = ahora_utc.strftime("%Y-%m-%d")
        hora_def = ahora_utc.strftime("%H:%M:%S")
        
        self.crear_widgets(lat_def, lon_def, fecha_def, hora_def)
        
        # Guardar configuración automáticamente al cerrar la ventana
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Guarda las coordenadas actuales y cierra la aplicación."""
        try:
            lat = self.vars["Latitud (°)"].get()
            lon = self.vars["Longitud (°)"].get()
            guardar_configuracion(lat, lon)
        except Exception:
            pass
        self.destroy()
        
    def ordenar_columna(self, col, reverse):
        lista_valores = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        try:
            lista_valores.sort(key=lambda t: float(re.sub(r'[^\d.-]', '', t[0])), reverse=reverse)
        except ValueError:
            lista_valores.sort(reverse=reverse)
        
        for indice, (valor, k) in enumerate(lista_valores):
            self.tree.move(k, "", indice)
        
        self.tree.heading(col, command=lambda: self.ordenar_columna(col, not reverse))

    def ordenar_optimo(self):
        items = self.tree.get_children("")
        if not items:
            return
        
        lista_datos = []
        for k in items:
            vals = self.tree.item(k, "values")
            try:
                rot = float(vals[5])  # Índice ajustado al eliminar B-V de pantalla
                horas = float(vals[6])
                ratio = (rot / horas) if horas > 0 else float('inf')
            except ValueError:
                ratio = float('inf')
            lista_datos.append((ratio, k))
        
        lista_datos.sort(key=lambda x: x[0])
        for indice, (ratio, k) in enumerate(lista_datos):
            self.tree.move(k, "", indice)
        
        self.actualizar_estado("Ordenado por criterio óptimo (Rotación / Horas Visibles).")

    def crear_widgets(self, lat_def, lon_def, fecha_def, hora_def):
        frame_form = ttk.LabelFrame(self, text="Parámetros de Observación")
        frame_form.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.vars = {}
        campos = [
            ("Latitud (°)", lat_def), ("Longitud (°)", lon_def), 
            ("Fecha (YYYY-MM-DD)", fecha_def), ("Hora Inicio (UT)", hora_def),
            ("Altura Mínima (°)", "30.0"),
            ("Mag. Mínima (Brillante)", "8.0"), ("Mag. Máxima (Débil)", "16.0"),
            ("Periodo Rot. Mín (h)", "2.0"), ("Periodo Rot. Máx (h)", "12.0")
        ]
        
        for i, (label_text, default) in enumerate(campos):
            ttk.Label(frame_form, text=label_text).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            var = tk.StringVar(value=default)
            self.vars[label_text] = var
            ttk.Entry(frame_form, textvariable=var, width=15).grid(row=i, column=1, padx=5, pady=5)
        
        self.btn_calcular = ttk.Button(frame_form, text="Calcular", command=self.ejecutar_calculo)
        self.btn_calcular.grid(row=len(campos), column=0, columnspan=2, pady=(15, 5))
        
        self.btn_optimo = ttk.Button(frame_form, text="Ordenar Óptimo (Rot / Vis)", command=self.ordenar_optimo)
        self.btn_optimo.grid(row=len(campos)+1, column=0, columnspan=2, pady=(0, 5))
        
        self.btn_sin_rotacion = ttk.Button(frame_form, text="Añadir sin Rotación", command=self.ejecutar_calculo_sin_rotacion)
        self.btn_sin_rotacion.grid(row=len(campos)+2, column=0, columnspan=2, pady=(0, 5))

        self.btn_astrometry = ttk.Button(frame_form, text="Subir a Astrometry.net", command=self.subir_astrometry)
        self.btn_astrometry.grid(row=len(campos)+3, column=0, columnspan=2, pady=(0, 10))
        
        self.lbl_estado = ttk.Label(frame_form, text="Esperando...", foreground="blue")
        self.lbl_estado.grid(row=len(campos)+4, column=0, columnspan=2)

        frame_tabla = ttk.Frame(self)
        frame_tabla.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Columna B-V eliminada de la visualización principal
        columnas = ("Asteroide", "AR (J2000)", "Dec (J2000)", "Altura Inicial", "Mag V", "Rotación (h)", "Horas Visibles")
        self.tree = ttk.Treeview(frame_tabla, columns=columnas, show="headings")
        
        for col in columnas:
            self.tree.heading(col, text=col, command=lambda c=col: self.ordenar_columna(c, False))
            self.tree.column(col, width=130, anchor="center")
        
        scrollbar = ttk.Scrollbar(frame_tabla, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.tree.bind("<Double-1>", self.on_asteroide_double_click)
        self.tree.bind("<Button-3>", self.mostrar_menu_contextual)
        self.tree.bind("<Button-2>", self.mostrar_menu_contextual)

    def actualizar_estado(self, texto):
        self.lbl_estado.config(text=texto)
        self.update_idletasks()

    def preparar_datos_base(self):
        if self.mpc_df is None:
            descargar_mpcorb()
            self.actualizar_estado("Cargando catálogo orbital MPCORB...")
            self.mpc_df = cargar_mpcorb()
            self.mpc_df['clean_des'] = self.mpc_df['designation'].apply(_jpl_pdes_from_mpc)
        
        if self.jpl_df is None:
            self.jpl_df = descargar_jpl_rotacion()

        if self.astorb_df is None:
            self.actualizar_estado("Cargando B-V desde ASTORB.DAT...")
            self.astorb_df = cargar_astorb()

    def ejecutar_calculo(self):
        self.tree.delete(*self.tree.get_children())
        self.asteroides_dict.clear()
        
        try:
            self.preparar_datos_base()
            self.actualizar_estado("Filtrando por rotación...")
            
            lat = float(self.vars["Latitud (°)"].get())
            lon = float(self.vars["Longitud (°)"].get())
            fecha = self.vars["Fecha (YYYY-MM-DD)"].get()
            hora = self.vars["Hora Inicio (UT)"].get()
            alt_min = float(self.vars["Altura Mínima (°)"].get())
            mag_min = float(self.vars["Mag. Mínima (Brillante)"].get())
            mag_max = float(self.vars["Mag. Máxima (Débil)"].get())
            rot_min = float(self.vars["Periodo Rot. Mín (h)"].get())
            rot_max = float(self.vars["Periodo Rot. Máx (h)"].get())
            
            df = pd.merge(self.mpc_df, self.jpl_df, left_on="clean_des", right_on="pdes", how="inner")
            df = df[(df['rot_per_h'] >= rot_min) & (df['rot_per_h'] <= rot_max)].copy()
            
            if df.empty:
                messagebox.showinfo("Resultado", "Ningún asteroide cumple el filtro de rotación.")
                self.actualizar_estado("Finalizado.")
                return

            df = pd.merge(df, self.astorb_df, left_on="clean_des", right_on="clean_key", how="left")

            self.actualizar_estado("Calculando posiciones iniciales...")
            t_inicio = Time(f"{fecha} {hora}")
            loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
            
            altitudes, magnitudes, ras, decs = calcular_posiciones(df, t_inicio, loc, return_eq=True)
            df["Alt_T0"], df["Mag_T0"], df["RA_deg"], df["DEC_deg"] = altitudes, magnitudes, ras, decs
            
            visibles = df[(df["Alt_T0"] >= alt_min) & (df["Mag_T0"] >= mag_min) & (df["Mag_T0"] <= mag_max)].copy()
            
            if visibles.empty:
                messagebox.showinfo("Resultado", "Ningún asteroide cumple los criterios de visibilidad inicial.")
                self.actualizar_estado("Finalizado.")
                return

            self.actualizar_estado("Proyectando visibilidad nocturna (Sol <= -12°)...")
            dt_base = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M:%S")
            intervalos_minutos = 15
            tiempos_noche = [dt_base + timedelta(minutes=i*intervalos_minutos) for i in range(48)]
            times_astropy = Time(tiempos_noche)
            
            # Cálculo de la altura del Sol para cada intervalo temporal (Crepúsculo náutico <= -12°)
            sun_altaz = get_sun(times_astropy).transform_to(AltAz(obstime=times_astropy, location=loc, pressure=0*u.hPa))
            sun_alts = sun_altaz.alt.deg

            horas_visibles_list = []
            for idx, row in visibles.iterrows():
                temp_df = pd.DataFrame([row] * len(times_astropy))
                altos, _ = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
                # Condición: Asteroide sobre altura mínima Y Sol a -12° o más bajo el horizonte
                tramos_visibles = np.sum((altos >= alt_min) & (sun_alts <= -12.0))
                horas_visibles_list.append((tramos_visibles * intervalos_minutos) / 60.0)
            
            visibles["Horas_Visibles"] = horas_visibles_list
            visibles = visibles.sort_values(by=["Horas_Visibles", "Mag_T0"], ascending=[False, True])
            
            self.actualizar_estado("Generando tabla final...")
            for _, row in visibles.iterrows():
                ar_angle = Angle(row['RA_deg'], u.deg)
                dec_angle = Angle(row['DEC_deg'], u.deg)
                
                ra_str = ar_angle.to_string(unit=u.hour, sep=':', precision=2, pad=True)
                dec_str = dec_angle.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
                dec_str = dec_str.replace(':', '°', 1).replace(':', "'", 1) + '"'
                ra_str = ra_str.replace(':', 'h', 1).replace(':', 'm', 1) + 's'
                
                valores = (
                    row["designation"], 
                    ra_str, 
                    dec_str, 
                    f"{row['Alt_T0']:.1f}°", 
                    f"{row['Mag_T0']:.2f}", 
                    f"{row['rot_per_h']:.2f}", 
                    f"{row['Horas_Visibles']:.1f}"
                )
                
                item_id = self.tree.insert("", tk.END, values=valores)
                self.asteroides_dict[item_id] = row.to_dict()
            
            self.actualizar_estado(f"¡Listo! Encontrados {len(visibles)} asteroides.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ha ocurrido un error:\n{str(e)}")
            self.actualizar_estado("Error en el cálculo.")

    def ejecutar_calculo_sin_rotacion(self):
        try:
            self.preparar_datos_base()
            self.actualizar_estado("Buscando asteroides sin periodo de rotación...")
            
            lat = float(self.vars["Latitud (°)"].get())
            lon = float(self.vars["Longitud (°)"].get())
            fecha = self.vars["Fecha (YYYY-MM-DD)"].get()
            hora = self.vars["Hora Inicio (UT)"].get()
            alt_min = float(self.vars["Altura Mínima (°)"].get())
            mag_min = float(self.vars["Mag. Mínima (Brillante)"].get())
            mag_max = float(self.vars["Mag. Máxima (Débil)"].get())
            
            df = pd.merge(self.mpc_df, self.jpl_df, left_on="clean_des", right_on="pdes", how="left")
            df = df[df['rot_per_h'].isna()].copy()
            
            if df.empty:
                messagebox.showinfo("Resultado", "No hay asteroides sin periodo de rotación en el catálogo.")
                self.actualizar_estado("Finalizado.")
                return

            df = pd.merge(df, self.astorb_df, left_on="clean_des", right_on="clean_key", how="left")

            self.actualizar_estado("Calculando posiciones iniciales...")
            t_inicio = Time(f"{fecha} {hora}")
            loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
            
            altitudes, magnitudes, ras, decs = calcular_posiciones(df, t_inicio, loc, return_eq=True)
            df["Alt_T0"], df["Mag_T0"], df["RA_deg"], df["DEC_deg"] = altitudes, magnitudes, ras, decs
            
            visibles = df[(df["Alt_T0"] >= alt_min) & (df["Mag_T0"] >= mag_min) & (df["Mag_T0"] <= mag_max)].copy()
            
            if visibles.empty:
                messagebox.showinfo("Resultado", "Ningún asteroide sin periodo cumple los criterios de visibilidad.")
                self.actualizar_estado("Finalizado.")
                return

            self.actualizar_estado("Proyectando visibilidad nocturna (Sol <= -12°)...")
            dt_base = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M:%S")
            intervalos_minutos = 15
            tiempos_noche = [dt_base + timedelta(minutes=i*intervalos_minutos) for i in range(48)]
            times_astropy = Time(tiempos_noche)
            
            # Cálculo de la altura del Sol para cada intervalo temporal (Crepúsculo náutico <= -12°)
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
            visibles["rot_per_h"] = np.nan
            
            self.actualizar_estado("Añadiendo al listado...")
            for _, row in visibles.iterrows():
                ar_angle = Angle(row['RA_deg'], u.deg)
                dec_angle = Angle(row['DEC_deg'], u.deg)
                
                ra_str = ar_angle.to_string(unit=u.hour, sep=':', precision=2, pad=True)
                dec_str = dec_angle.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
                dec_str = dec_str.replace(':', '°', 1).replace(':', "'", 1) + '"'
                ra_str = ra_str.replace(':', 'h', 1).replace(':', 'm', 1) + 's'
                
                valores = (
                    row["designation"], 
                    ra_str, 
                    dec_str, 
                    f"{row['Alt_T0']:.1f}°", 
                    f"{row['Mag_T0']:.2f}", 
                    "Desconocido", 
                    f"{row['Horas_Visibles']:.1f}"
                )
                
                item_id = self.tree.insert("", tk.END, values=valores)
                self.asteroides_dict[item_id] = row.to_dict()
            
            self.actualizar_estado(f"¡Listo! Añadidos {len(visibles)} sin periodo.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ha ocurrido un error:\n{str(e)}")
            self.actualizar_estado("Error en el cálculo.")

    def subir_astrometry(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen para Astrometry.net",
            filetypes=[("Imágenes y FITS", "*.jpg *.jpeg *.png *.fits *.fit"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            webbrowser.open("https://nova.astrometry.net/upload")
            messagebox.showinfo(
                "Astrometry.net",
                f"Imagen seleccionada:\n{os.path.basename(file_path)}\n\n"
                "Se va a abrir la página de subida de Astrometry.net en tu navegador."
            )

    def mostrar_menu_contextual(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="📈 Curva (JPL)", command=lambda: self.abrir_curva(item))
            menu.add_command(label="🧊 Ver modelo 3D en DAMIT", command=lambda: self.abrir_damit(item))
            menu.add_command(label="🎯 Generar VOTable de comparación (APASS + VSX)", command=lambda: self.generar_votable_comparacion(item))
            menu.add_command(label="🌌 Aladin Lite", command=lambda: self.abrir_aladin(item))
            menu.add_command(label="📋 Ver tabla de trayectoria horaria", command=lambda: self.mostrar_tabla_trayectoria(item))
            menu.add_command(label="📊 Gráfica de Altitud (Noche)", command=lambda: self.graficar_altitud(item))
            menu.add_command(label="📉 Variación de Magnitud (30 días)", command=lambda: self.graficar_magnitud(item))
            menu.add_command(label="📖 Buscar en MPB (Lightcurve Index)", command=lambda: self.buscar_en_mpb(item))
            menu.add_command(label="🌐 Buscar en ALCDEF", command=lambda: self.abrir_alcdef(item))
            menu.tk_popup(event.x_root, event.y_root)

    def abrir_damit(self, item):
        row_dict = self.asteroides_dict.get(item)
        if row_dict:
            pdes = row_dict.get("clean_des", row_dict["designation"])
            url = f"https://damit.cuni.cz/projects/damit/?q={pdes}"
            webbrowser.open(url)

    def generar_votable_comparacion(self, item):
        row_dict = self.asteroides_dict.get(item)
        if not row_dict:
            return

        nombre = row_dict["designation"]
        ra_ast = row_dict["RA_deg"]
        dec_ast = row_dict["DEC_deg"]
        mag_ast = row_dict["Mag_T0"]

        output_filename = filedialog.asksaveasfilename(
            defaultextension=".vot",
            filetypes=[("VOTable files", "*.vot"), ("All files", "*.*")],
            title=f"Guardar VOTable de comparación para {nombre}",
            initialfile=f"comparacion_{re.sub(r'[^a-zA-Z0-9]', '_', nombre)}.vot"
        )
        if not output_filename:
            return

        try:
            self.actualizar_estado(f"Conectando con VizieR para APASS (radio 1° en {nombre})...")
            vizier = Vizier(row_limit=-1, columns=['*', 'Bmag', 'Vmag', 'RAJ2000', 'DEJ2000'])
            coord = SkyCoord(ra=ra_ast, dec=dec_ast, unit=(u.deg, u.deg), frame='icrs')
            
            resultados = vizier.query_region(coord, radius=1.0 * u.deg, catalog="II/336/apass9")
            if not resultados:
                messagebox.showwarning("Aviso", "No se encontraron estrellas en APASS DR9 para esta región.")
                self.actualizar_estado("Finalizado.")
                return

            tabla = resultados[0]
            tabla['B_minus_V'] = tabla['Bmag'] - tabla['Vmag']

            mag_min = mag_ast - 1.0
            mag_max = mag_ast + 1.0
            filtro_mag = (tabla['Vmag'] >= mag_min) & (tabla['Vmag'] <= mag_max)
            filtro_bv = (tabla['B_minus_V'] >= 0.6) & (tabla['B_minus_V'] <= 0.9)
            tabla_candidata = tabla[filtro_mag & filtro_bv]

            if len(tabla_candidata) == 0:
                messagebox.showwarning("Aviso", "Ninguna estrella cumple los rangos de magnitud y B-V especificados.")
                self.actualizar_estado("Finalizado.")
                return

            self.actualizar_estado("Consultando catálogo VSX para exclusión de variables...")
            v_vsx = Vizier(columns=['OID', 'Name', 'Type', '_RAJ2000', '_DEJ2000'], row_limit=-1)
            res_vsx = v_vsx.query_region(coord, radius=1.0 * u.deg, catalog='B/vsx/vsx')

            if res_vsx and len(res_vsx) > 0:
                tabla_vsx = res_vsx[0]
                coor_candidatas = SkyCoord(ra=tabla_candidata['RAJ2000'], dec=tabla_candidata['DEJ2000'], unit=(u.deg, u.deg), frame='icrs')
                coor_vsx = SkyCoord(ra=tabla_vsx['_RAJ2000'], dec=tabla_vsx['_DEJ2000'], frame='icrs')
                
                _, sep, _ = coor_candidatas.match_to_catalog_sky(coor_vsx)
                keep_mask = sep.to(u.arcsec).value > 3.0
                tabla_final = tabla_candidata[keep_mask]
            else:
                tabla_final = tabla_candidata

            if len(tabla_final) == 0:
                messagebox.showwarning("Aviso", "Tras excluir las estrellas variables de VSX, no quedan candidatas.")
                self.actualizar_estado("Finalizado.")
                return

            tabla_final['RAJ2000'].ucd = "pos.eq.ra;meta.main"
            tabla_final['DEJ2000'].ucd = "pos.eq.dec;meta.main"

            votable = from_table(tabla_final)
            writeto(votable, output_filename)
            self.actualizar_estado("Fichero VOTable generado con éxito.")
            messagebox.showinfo("Éxito", f"Archivo VOTable guardado correctamente:\n{output_filename}\n\nEstrellas de comparación no variables: {len(tabla_final)}")

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al generar el VOTable:\n{str(e)}")
            self.actualizar_estado("Error en la exportación.")

    def graficar_altitud(self, item):
        row_dict = self.asteroides_dict.get(item)
        if not row_dict:
            return
        
        nombre = row_dict["designation"]
        lat = float(self.vars["Latitud (°)"].get())
        lon = float(self.vars["Longitud (°)"].get())
        fecha = self.vars["Fecha (YYYY-MM-DD)"].get()
        hora = self.vars["Hora Inicio (UT)"].get()
        alt_min = float(self.vars["Altura Mínima (°)"].get())
        
        dt_base = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M:%S")
        intervalos_minutos = 15
        tiempos = [dt_base + timedelta(minutes=i*intervalos_minutos) for i in range(49)]
        times_astropy = Time(tiempos)
        
        temp_df = pd.DataFrame([row_dict] * len(times_astropy))
        loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
        altitudes, _ = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
        
        # Cálculo de la altura del Sol para la misma franja horaria
        sun_altaz = get_sun(times_astropy).transform_to(AltAz(obstime=times_astropy, location=loc, pressure=0*u.hPa))
        sun_altitudes = sun_altaz.alt.deg

        etiquetas_tiempo = [t.strftime("%H:%M") for t in tiempos]
        
        plt.figure(figsize=(9, 5))
        # Curva de altura del asteroide
        plt.plot(etiquetas_tiempo, altitudes, marker='o', color='dodgerblue', linewidth=2, label='Altura asteroide')
        # Curva de altura del Sol
        plt.plot(etiquetas_tiempo, sun_altitudes, marker='x', color='gold', linewidth=1.5, linestyle='-.', label='Altura Sol')
        
        # Líneas de referencia
        plt.axhline(y=alt_min, color='crimson', linestyle='--', linewidth=1.5, label=f'Altura mínima asteroide ({alt_min}°)')
        plt.axhline(y=-12.0, color='darkorange', linestyle='-', linewidth=1.5, label='Crepúsculo náutico (Sol -12°)')

        plt.title(f"Evolución de Altura (Asteroide y Sol) - {nombre}", fontsize=12, fontweight='bold')
        plt.xlabel("Hora UT", fontsize=10)
        plt.ylabel("Altura (°)", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(rotation=45, fontsize=8)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def graficar_magnitud(self, item):
        row_dict = self.asteroides_dict.get(item)
        if not row_dict:
            return
        
        nombre = row_dict["designation"]
        lat = float(self.vars["Latitud (°)"].get())
        lon = float(self.vars["Longitud (°)"].get())
        fecha_str = self.vars["Fecha (YYYY-MM-DD)"].get()
        hora = self.vars["Hora Inicio (UT)"].get()
        
        fecha_base = datetime.strptime(fecha_str, "%Y-%m-%d").date()
        fechas = [fecha_base + timedelta(days=i) for i in range(30)]
        tiempos = [datetime.combine(f, datetime.strptime(hora, "%H:%M:%S").time()) for f in fechas]
        times_astropy = Time(tiempos)
        
        temp_df = pd.DataFrame([row_dict] * len(times_astropy))
        loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
        _, magnitudes = calcular_posiciones(temp_df, times_astropy, loc, return_eq=False)
        
        etiquetas_fechas = [f.strftime("%d/%m") for f in fechas]
        
        plt.figure(figsize=(10, 5))
        plt.plot(etiquetas_fechas, magnitudes, marker='s', color='forestgreen', linewidth=2, label='Magnitud V estimada')
        plt.gca().invert_yaxis()
        plt.title(f"Variación de Magnitud (Próximos 30 días) - {nombre}", fontsize=12, fontweight='bold')
        plt.xlabel("Fecha", fontsize=10)
        plt.ylabel("Magnitud V (más brillo ↑)", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(rotation=45, fontsize=8)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def abrir_aladin(self, item):
        row_dict = self.asteroides_dict.get(item)
        if row_dict:
            ra = row_dict["RA_deg"]
            dec = row_dict["DEC_deg"]
            url = f"https://aladin.cds.unistra.fr/AladinLite/?target={ra}+{dec}"
            webbrowser.open(url)

    def mostrar_tabla_trayectoria(self, item):
        row_dict = self.asteroides_dict.get(item)
        if not row_dict:
            return
        
        nombre = row_dict["designation"]
        lat = float(self.vars["Latitud (°)"].get())
        lon = float(self.vars["Longitud (°)"].get())
        fecha = self.vars["Fecha (YYYY-MM-DD)"].get()
        hora = self.vars["Hora Inicio (UT)"].get()
        
        dt_base = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M:%S")
        tiempos = [dt_base + timedelta(hours=i) for i in range(13)]
        times_astropy = Time(tiempos)
        
        temp_df = pd.DataFrame([row_dict] * len(times_astropy))
        loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=10*u.m)
        altitudes, magnitudes, ras, decs = calcular_posiciones(temp_df, times_astropy, loc, return_eq=True)
        
        win = tk.Toplevel(self)
        win.title(f"Trayectoria Horaria - {nombre}")
        win.geometry("700x400")
        
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columnas = ("Hora (UT)", "AR (J2000)", "Dec (J2000)", "Altura", "Mag V")
        tree = ttk.Treeview(frame, columns=columnas, show="headings")
        
        for col in columnas:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        for t, ra_val, dec_val, alt, mag in zip(tiempos, ras, decs, altitudes, magnitudes):
            ar_angle = Angle(ra_val, u.deg)
            dec_angle = Angle(dec_val, u.deg)
            
            ra_str = ar_angle.to_string(unit=u.hour, sep=':', precision=2, pad=True)
            dec_str = dec_angle.to_string(unit=u.degree, sep=':', precision=1, alwayssign=True, pad=True)
            dec_str = dec_str.replace(':', '°', 1).replace(':', "'", 1) + '"'
            ra_str = ra_str.replace(':', 'h', 1).replace(':', 'm', 1) + 's'
            
            tree.insert("", tk.END, values=(t.strftime("%H:%M UT"), ra_str, dec_str, f"{alt:.1f}°", f"{mag:.2f}"))
        
        btn_copiar = ttk.Button(win, text="Copiar tabla al portapapeles", command=lambda: self.copiar_tabla_portapapeles(tree))
        btn_copiar.pack(pady=10)

    def copiar_tabla_portapapeles(self, tree):
        texto = ""
        for item in tree.get_children():
            vals = tree.item(item, "values")
            texto += "\t".join(vals) + "\n"
        self.clipboard_clear()
        self.clipboard_append(texto)
        messagebox.showinfo("Copiado", "Trayectoria copiada al portapapeles.")

    def abrir_alcdef(self, item):
        row_dict = self.asteroides_dict.get(item)
        if row_dict:
            nombre = row_dict["designation"]
            self.clipboard_clear()
            self.clipboard_append(nombre)
            webbrowser.open("https://alcdef.org")

    def abrir_curva(self, item):
        row_dict = self.asteroides_dict.get(item)
        if not row_dict:
            return
        
        nombre = row_dict["designation"]
        pdes = row_dict["clean_des"]
        rot_per = row_dict.get("rot_per_h", np.nan)
        rot_str = f"{rot_per:.4f} horas" if pd.notna(rot_per) else "Desconocido (No registrado)"
        
        win = tk.Toplevel(self)
        win.title(f"Curva y Parámetros - {nombre}")
        win.geometry("450x300")
        win.resizable(False, False)
        
        ttk.Label(win, text=f"Asteroide: {nombre}", font=("Arial", 12, "bold")).pack(pady=15)
        ttk.Label(win, text=f"Periodo de rotación registrado: {rot_str}", font=("Arial", 10)).pack(pady=5)
        ttk.Label(win, text="Para consultar la curva de luz oficial, fotometría\ny observaciones guardadas en el JPL SBDB:", justify=tk.CENTER).pack(pady=15)
        
        url_jpl = f"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={pdes}"
        btn_jpl = ttk.Button(win, text="Abrir página de Curva / JPL SBDB", command=lambda: webbrowser.open(url_jpl))
        btn_jpl.pack(pady=10)
        ttk.Button(win, text="Cerrar", command=win.destroy).pack(pady=10)

    def buscar_en_mpb(self, item):
        if not PYPDF_AVAILABLE:
            messagebox.showerror("Error", "La librería 'pypdf' no está instalada.\nInstálala desde Anaconda Prompt con: conda install pypdf")
            return
        
        row_dict = self.asteroides_dict.get(item)
        if not row_dict:
            return
        
        nombre = row_dict["designation"]
        pdes = row_dict["clean_des"]
        
        if not os.path.exists(MPB_PDF_PATH):
            self.actualizar_estado("Descargando índice PDF de MPB...")
            try:
                url_pdf = "https://mpbulletin.org/index/MPB_LightcurveIndex.pdf"
                r = requests.get(url_pdf, timeout=30)
                r.raise_for_status()
                with open(MPB_PDF_PATH, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo descargar el PDF del MPB:\n{str(e)}")
                return

        self.actualizar_estado("Buscando en el índice del MPB...")
        
        coincidencias = []
        try:
            reader = PdfReader(MPB_PDF_PATH)
            termino = pdes.strip().upper()
            
            for page in reader.pages:
                texto = page.extract_text()
                if texto:
                    for linea in texto.split("\n"):
                        if re.search(r'\b' + re.escape(termino) + r'\b', linea, re.IGNORECASE):
                            coincidencias.append(linea.strip())
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar el fichero PDF:\n{str(e)}")
            return

        self.actualizar_estado("Búsqueda en MPB finalizada.")

        if not coincidencias:
            messagebox.showinfo("MPB Index", f"No se han encontrado referencias directas para '{nombre}' (ID: {pdes}) en el índice del MPB.")
            return

        resultado_str = "\n".join(coincidencias[:15])
        if len(coincidencias) > 15:
            resultado_str += f"\n\n... y {len(coincidencias) - 15} referencias más."

        msg = f"Se han encontrado {len(coincidencias)} referencias en el MPB para {nombre}:\n\n{resultado_str}\n\n¿Desea abrir la página oficial de búsqueda del MPB para ver los detalles?"
        
        if messagebox.askyesno("Referencias encontradas en MPB", msg):
            url_busqueda = "https://mpbulletin.org/index.php?searchFirst=&searchOptions1=abstracts&searchBoolean=AND&searchSecond=&searchOptions2=authors&searchYear1=1973&searchYear2=2026&doSearch=Submit&task=doSearch"
            webbrowser.open(url_busqueda)

    def on_asteroide_double_click(self, event):
        seleccion = self.tree.selection()
        if seleccion:
            self.abrir_curva(seleccion[0])

if __name__ == "__main__":
    app = AppAsteroides()
    app.mainloop()