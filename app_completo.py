"""
Interfaz web para el modelo de dotación SCEL (ELI148).

Ejecutar : python -m streamlit run app.py
Requiere : pip install "pulp==2.9.0" pandas streamlit

Reutiliza las funciones de Modelo_Avance_GrupoXX.py; los supuestos se
ajustan desde la barra lateral sin editar el código.
"""
import copy
import importlib.util
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import Modelo_Avance_GrupoXX as M

st.set_page_config(page_title="Dotación SCEL", page_icon="✈️", layout="wide")

# Valores originales del modelo. Se guardan una sola vez en el módulo porque
# Streamlit re-ejecuta este script y abajo se sobrescriben los globales de M.
if not hasattr(M, "_ORIGINALES"):
    M._ORIGINALES = {k: copy.deepcopy(getattr(M, k)) for k in
                     ("USO_MESON", "PROD_BASE", "AGENTES_POR_VUELO", "CAPACIDAD",
                      "COSTO_HORA", "PENALIZACION", "ALFA", "W_MAX")}
DEFAULTS = M._ORIGINALES
H24 = list(range(24))
HAY_MPL = importlib.util.find_spec("matplotlib") is not None  # para colorear tablas


@st.cache_data(show_spinner="Cargando datos de la JAC…")
def cargar():
    return M.cargar_embarcados()


@st.cache_data
def pps_mes(anio_mes):
    return M.pax_por_salida(cargar(), anio_mes)


@st.cache_data
def pax_diarios():
    e = cargar()
    return e.groupby(["Fecha", "ruta"])["pax"].sum().unstack(fill_value=0)


emb = cargar()
FMIN, FMAX = emb["Fecha"].min().date(), emb["Fecha"].max().date()

# ---------------------------------------------------------------------------
# Barra lateral: supuestos
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.title("⚙️ Supuestos")
if sb.button("Restaurar valores originales"):
    for k in list(st.session_state):
        if k.startswith("p_"):
            del st.session_state[k]
    st.rerun()

with sb.expander("Nivel de servicio y costos", expanded=True):
    alfa = st.slider("α (nivel mínimo de servicio)", 0.0, 1.0, DEFAULTS["ALFA"], 0.01, key="p_alfa")
    W = st.number_input("W (personal disponible por día)", 1, 100000, DEFAULTS["W_MAX"], 10, key="p_W")
    P = st.number_input("Penalización por pax no atendido (CLP)", 0, 1_000_000,
                        DEFAULTS["PENALIZACION"], 100, key="p_P")
    c_ft = st.number_input("Costo hora FT (CLP)", 0, 1_000_000, DEFAULTS["COSTO_HORA"]["FT"], 500, key="p_cft")
    c_pt = st.number_input("Costo hora PT (CLP)", 0, 1_000_000, DEFAULTS["COSTO_HORA"]["PT"], 500, key="p_cpt")

with sb.expander("Productividad y uso de mesón"):
    prod_ci = st.number_input("Check-in (pax/trabajador·h)", 1, 1000, DEFAULTS["PROD_BASE"]["Checkin"], key="p_pci")
    prod_se = st.number_input("Seguridad (pax/trabajador·h)", 1, 1000, DEFAULTS["PROD_BASE"]["Seguridad"], key="p_pse")
    agentes = st.number_input("Agentes por vuelo (embarque)", 1, 20, DEFAULTS["AGENTES_POR_VUELO"], key="p_ag")
    uso_nac = st.slider("Uso de mesón NAC", 0.0, 1.0, DEFAULTS["USO_MESON"]["NAC"], 0.05, key="p_unac")
    uso_int = st.slider("Uso de mesón INT", 0.0, 1.0, DEFAULTS["USO_MESON"]["INT"], 0.05, key="p_uint")
    meses = sorted({int(f"{d.year}{d.month:02d}") for d in pd.date_range(FMIN, FMAX, freq="MS")})
    mes_pps = st.selectbox("Mes para pax por despegue", meses, index=meses.index(202509), key="p_mes")

with sb.expander("Capacidad física (puestos)"):
    cap = {a: st.number_input(a, 0, 1000, DEFAULTS["CAPACIDAD"][a], key=f"p_cap_{a}") for a in M.AREAS}

with sb.expander("Turnos y solver"):
    largo_ft = st.number_input("Largo turno FT (h)", 1, 24, 8, key="p_lft")
    largo_pt = st.number_input("Largo turno PT (h)", 1, 24, 4, key="p_lpt")
    tmax = st.number_input("Tiempo máximo del solver (s)", 5, 3600, 120, 5, key="p_tmax")

# Se escriben los supuestos en el módulo, ya que sus funciones los leen como globales.
M.ALFA, M.W_MAX, M.PENALIZACION = alfa, W, P
M.COSTO_HORA = {"FT": c_ft, "PT": c_pt}
M.PROD_BASE = {"Checkin": prod_ci, "Seguridad": prod_se}
M.AGENTES_POR_VUELO = agentes
M.USO_MESON = {"NAC": uso_nac, "INT": uso_int}
M.CAPACIDAD = cap
pps = pps_mes(mes_pps)


def pats_n(n_horas):
    return M.patrones_turno(n_horas, largo_ft, largo_pt)


def resolver_dia(fecha, f_dem=1.0, f_prod=1.0):
    sal = M.salidas_del_dia(emb, fecha)
    D = M.demanda_por_area(sal, H24, f_dem)
    mu = M.productividad(pps, f_prod)
    pats = pats_n(24)
    r = M.resolver(M.AREAS, H24, D, mu, cap, pats, W, alfa, P, tmax)
    r["pax_dia"] = f_dem * (sum(sal["NAC"].values()) + sum(sal["INT"].values()))
    if r["estado"] == "Optimal":
        r["verif"] = M.verificar(r, M.AREAS, H24, D, mu, cap, pats, W, alfa)
    else:
        r_sin = M.resolver(M.AREAS, H24, D, mu, cap, pats, 10**6, alfa, P, tmax)
        if r_sin["estado"] == "Optimal":
            r["diag"] = (f"Con W={W} no alcanza el personal: se necesitan "
                         f"{r_sin['trabajadores']} personas (costo ${r_sin['objetivo']:,.0f}).")
        else:
            falta = [a for a in M.AREAS
                     if any((1 - alfa) * D[a][t] < D[a][t] - mu[a] * cap[a] for t in H24)]
            r["diag"] = "Falta capacidad física (puestos) en: " + ", ".join(falta)
    return r, D, mu, sal


def fila_resumen(nombre, r):
    ok = r["estado"] == "Optimal"
    return {"Escenario": nombre, "Pax del día": round(r.get("pax_dia", 0)), "Estado": r["estado"],
            "Costo (CLP)": round(r["objetivo"]) if ok else None,
            "Personal": r["trabajadores"] if ok else None,
            "No atendidos (pax)": round(r["deficit_total"], 1) if ok else None,
            "% no atendidos": round(100 * r["deficit_total"] / r["demanda_total"], 2)
            if ok and r["demanda_total"] else None,
            "Tiempo (s)": round(r["tiempo_s"], 2), "Nota": r.get("diag", "")}


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------
st.title("✈️ Dotación de personal — Aeropuerto SCEL")
st.caption(f"Datos reales JAC del {FMIN:%d-%m-%Y} al {FMAX:%d-%m-%Y} · "
           f"pax por despegue ({mes_pps}): NAC={pps['NAC']:.0f}, INT={pps['INT']:.0f}")

tab_dia, tab_esc, tab_esc2, tab_datos, tab_ver = st.tabs(
    ["📅 Resolver un día", "📊 Comparar escenarios", "📈 Escalabilidad", "🗂️ Datos", "✅ Instancia de verificación"])

# --- Un día ----------------------------------------------------------------
with tab_dia:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    fecha = c1.date_input("Día a resolver", date(2025, 9, 2), FMIN, FMAX, format="DD/MM/YYYY")
    f_dem = c2.number_input("Factor de demanda", 0.1, 5.0, 1.0, 0.05)
    f_prod = c3.number_input("Factor de productividad", 0.1, 5.0, 1.0, 0.05)
    c4.write("")
    c4.write("")
    if c4.button("▶ Resolver", type="primary", width="stretch"):
        with st.spinner("Resolviendo MILP…"):
            st.session_state["dia"] = (fecha, *resolver_dia(fecha, f_dem, f_prod))

    if "dia" in st.session_state:
        f, r, D, mu, sal = st.session_state["dia"]
        st.subheader(f"Resultado para {f:%A %d-%m-%Y}")
        if r["estado"] != "Optimal":
            st.error(f"Estado: **{r['estado']}**. {r.get('diag', '')}")
        else:
            k = st.columns(6)
            k[0].metric("Costo total", f"${r['objetivo']:,.0f}")
            k[1].metric("Personal", f"{r['trabajadores']} / {W}")
            k[2].metric("Pax del día", f"{r['pax_dia']:,.0f}")
            k[3].metric("No atendidos", f"{r['deficit_total']:,.1f}",
                        f"{100 * r['deficit_total'] / r['demanda_total']:.2f} %", delta_color="inverse")
            k[4].metric("Tiempo", f"{r['tiempo_s']:.2f} s")
            k[5].metric("Variables / Restr.", f"{r['n_var']} / {r['n_res']}")
            if r["verif"] == "OK":
                st.success("Verificación de restricciones R1–R5: OK")
            else:
                st.warning(f"Restricciones violadas: {r['verif']}")

            filas = [{"Área": a, "Hora": t, "Demanda (pax/h)": D[a][t],
                      "Capacidad atendida (pax/h)": mu[a] * r["x"][a, t],
                      "Trabajadores": r["x"][a, t], "No atendidos": r["d"][a, t]}
                     for a in M.AREAS for t in H24]
            df = pd.DataFrame(filas)

            st.markdown("#### Demanda vs. capacidad por área")
            base = alt.Chart(df).encode(x=alt.X("Hora:O", title="Hora"))
            graf = alt.layer(
                base.mark_bar(opacity=0.5).encode(
                    y=alt.Y("Capacidad atendida (pax/h):Q", title="pax/h"),
                    tooltip=["Hora", "Trabajadores", alt.Tooltip("Capacidad atendida (pax/h):Q", format=".0f"),
                             alt.Tooltip("Demanda (pax/h):Q", format=".0f"),
                             alt.Tooltip("No atendidos:Q", format=".1f")]),
                base.mark_line(point=True, color="#d62728").encode(y="Demanda (pax/h):Q"),
            ).properties(width=380, height=200).facet(facet="Área:N", columns=3).resolve_scale(y="independent")
            st.altair_chart(graf)
            st.caption("Barras: capacidad = productividad × trabajadores · Línea roja: demanda")

            st.markdown("#### Trabajadores en servicio por hora")
            tabla_x = df.pivot(index="Área", columns="Hora", values="Trabajadores").reindex(M.AREAS)
            st.dataframe(tabla_x.style.background_gradient(cmap="Blues", axis=None) if HAY_MPL else tabla_x,
                         width="stretch")

            st.markdown("#### Turnos asignados")
            turnos = pd.DataFrame([{"Área": a, "Tipo": s[:2], "Inicio": int(s[2:]) % 24,
                                    "Fin": (int(s[2:]) + (largo_ft if s[:2] == "FT" else largo_pt)) % 24,
                                    "Personas": n} for (a, s), n in r["z"].items()])
            turnos = turnos.sort_values(["Área", "Inicio", "Tipo"])
            turnos["Inicio"] = turnos["Inicio"].map("{:02d}:00".format)
            turnos["Fin"] = turnos["Fin"].map("{:02d}:00".format)
            cc1, cc2 = st.columns([2, 1])
            cc1.dataframe(turnos, hide_index=True, width="stretch")
            cc2.dataframe(turnos.groupby("Tipo")["Personas"].sum().rename("Total"), width="stretch")

            st.markdown("#### Pasajeros embarcados por hora (dato real)")
            st.bar_chart(pd.DataFrame({"Nacional": sal["NAC"], "Internacional": sal["INT"]}))

            d1, d2 = st.columns(2)
            d1.download_button("⬇ Descargar detalle horario (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                               f"detalle_{f}.csv", "text/csv")
            d2.download_button("⬇ Descargar turnos (CSV)", turnos.to_csv(index=False).encode("utf-8-sig"),
                               f"turnos_{f}.csv", "text/csv")
    else:
        st.info("Elige un día y presiona **Resolver**.")

# --- Escenarios --------------------------------------------------------------
with tab_esc:
    st.markdown("Edita la tabla (agrega o borra filas) y presiona **Resolver escenarios**.")
    if "esc" not in st.session_state:
        st.session_state["esc"] = pd.DataFrame([
            {"Nombre": "Típico mar 02-09-2025", "Fecha": date(2025, 9, 2), "Factor demanda": 1.0, "Factor productividad": 1.0},
            {"Nombre": "Peak lun 02-02-2026", "Fecha": date(2026, 2, 2), "Factor demanda": 1.0, "Factor productividad": 1.0},
            {"Nombre": "Bajo sáb 23-05-2026", "Fecha": date(2026, 5, 23), "Factor demanda": 1.0, "Factor productividad": 1.0},
            {"Nombre": "Típico, productividad -20 %", "Fecha": date(2025, 9, 2), "Factor demanda": 1.0, "Factor productividad": 0.8},
        ])
    esc = st.data_editor(st.session_state["esc"], num_rows="dynamic", width="stretch", column_config={
        "Fecha": st.column_config.DateColumn(min_value=FMIN, max_value=FMAX, format="DD/MM/YYYY"),
        "Factor demanda": st.column_config.NumberColumn(min_value=0.1, max_value=5.0, step=0.05),
        "Factor productividad": st.column_config.NumberColumn(min_value=0.1, max_value=5.0, step=0.05)})
    if st.button("▶ Resolver escenarios", type="primary"):
        filas, barra = [], st.progress(0.0)
        validas = esc.dropna(subset=["Fecha"])
        for i, row in enumerate(validas.itertuples(index=False)):
            r, *_ = resolver_dia(row.Fecha, row[2] or 1.0, row[3] or 1.0)
            filas.append(fila_resumen(row.Nombre or str(row.Fecha), r))
            barra.progress((i + 1) / len(validas))
        st.session_state["esc_res"] = pd.DataFrame(filas)
    if "esc_res" in st.session_state:
        res = st.session_state["esc_res"]
        st.dataframe(res, hide_index=True, width="stretch",
                     column_config={"Costo (CLP)": st.column_config.NumberColumn(format="$%d")})
        ok = res.dropna(subset=["Costo (CLP)"])
        if len(ok):
            st.bar_chart(ok.set_index("Escenario")[["Costo (CLP)"]], horizontal=True)

# --- Escalabilidad -------------------------------------------------------------
with tab_esc2:
    c1, c2 = st.columns(2)
    inicio = c1.date_input("Día inicial", date(2025, 9, 1), FMIN, FMAX, format="DD/MM/YYYY", key="esc_ini")
    dias = c2.multiselect("Horizontes (días consecutivos)", [1, 2, 3, 5, 7, 10, 14, 21, 28], [1, 3, 7, 14])
    st.caption("Horizontes largos pueden tardar varios minutos (límite por corrida en la barra lateral).")
    if st.button("▶ Correr escalabilidad", type="primary"):
        filas, barra = [], st.progress(0.0)
        mu = M.productividad(pps)
        for i, n in enumerate(sorted(dias)):
            fechas = pd.date_range(inicio, periods=n)
            if fechas[-1].date() > FMAX:
                st.warning(f"{n} días excede el rango de datos; se omite.")
                continue
            Hn = list(range(24 * n))
            Dn = {a: {} for a in M.AREAS}
            for j, f in enumerate(fechas):
                Dd = M.demanda_por_area(M.salidas_del_dia(emb, f), H24)
                for a in M.AREAS:
                    for t in H24:
                        Dn[a][24 * j + t] = Dd[a][t]
            r = M.resolver(M.AREAS, Hn, Dn, mu, cap, pats_n(24 * n), W * n, alfa, P, tmax)
            filas.append({"Días": n, "Horas": len(Hn), "Variables": r["n_var"], "Restricciones": r["n_res"],
                          "Estado": r["estado"], "Tiempo (s)": round(r["tiempo_s"], 2),
                          "Costo (CLP)": round(r["objetivo"]) if r["estado"] == "Optimal" else None})
            barra.progress((i + 1) / len(dias))
        st.session_state["escala"] = pd.DataFrame(filas)
    if "escala" in st.session_state and len(st.session_state["escala"]):
        sc = st.session_state["escala"]
        st.dataframe(sc, hide_index=True, width="stretch")
        st.line_chart(sc.set_index("Variables")[["Tiempo (s)"]])

# --- Datos ---------------------------------------------------------------------
with tab_datos:
    c1, c2, c3 = st.columns(3)
    anio = c1.number_input("Año a validar", 2019, FMAX.year, 2025)
    mes = c2.number_input("Mes a validar", 1, 12, 1)
    try:
        e_, t_, err_ = M.validar_con_trafico(emb, int(anio), int(mes))
        c3.metric("Horario vs. tráfico mensual", f"{e_:,.0f} vs {t_:,.0f}", f"{100 * err_:+.2f} %")
    except Exception as ex:  # mes sin datos en trafico.csv
        c3.warning(f"No se pudo validar: {ex}")
    a25 = emb[emb.Fecha.dt.year == 2025]
    st.metric("Factor de ocupación 2025", f"{a25.pax.sum() / a25.asientos.sum():.1%}")
    st.markdown("#### Pasajeros embarcados por día (útil para elegir escenarios)")
    pd_ = pax_diarios()
    rango = st.slider("Rango", FMIN, FMAX, (date(2025, 1, 1), FMAX), format="DD/MM/YYYY")
    sub = pd_.loc[pd.Timestamp(rango[0]):pd.Timestamp(rango[1])]
    st.line_chart(sub)
    tot = sub.sum(axis=1)
    if len(tot):
        st.write(f"Día peak: **{tot.idxmax():%a %d-%m-%Y}** ({tot.max():,.0f} pax) · "
                 f"Día más bajo: **{tot.idxmin():%a %d-%m-%Y}** ({tot.min():,.0f} pax)")

# --- Instancia de verificación -----------------------------------------------
with tab_ver:
    st.markdown("Instancia pequeña (1 área, 6 h, 3 turnos) para comprobar el modelo a mano.")
    dem = st.data_editor(pd.DataFrame({"Hora": range(6), "Demanda": [100, 150, 200, 150, 100, 50]}),
                         hide_index=True, disabled=["Hora"])
    mu1 = st.number_input("Productividad (pax/trabajador·h)", 1, 1000, 25)
    if st.button("▶ Resolver instancia 1", type="primary"):
        H1 = list(range(6))
        D1 = {"Checkin_NAC": dict(zip(H1, dem["Demanda"].astype(float)))}
        pats1 = {"T06": {"horas": {0, 1, 2, 3}, "costo": 36000},
                 "T08": {"horas": {2, 3, 4, 5}, "costo": 36000},
                 "T06L": {"horas": {0, 1, 2, 3, 4, 5}, "costo": 50000}}
        r1 = M.resolver(["Checkin_NAC"], H1, D1, {"Checkin_NAC": mu1}, cap, pats1, W, alfa, P, tmax)
        if r1["estado"] == "Optimal":
            st.success(f"Óptimo: ${r1['objetivo']:,.0f} · turnos {dict((s, n) for (_, s), n in r1['z'].items())} · "
                       f"verificación: {M.verificar(r1, ['Checkin_NAC'], H1, D1, {'Checkin_NAC': mu1}, cap, pats1, W, alfa)}")
            st.dataframe(pd.DataFrame({"Hora": H1, "Trabajadores": [r1["x"]["Checkin_NAC", t] for t in H1],
                                       "No atendidos": [r1["d"]["Checkin_NAC", t] for t in H1]}), hide_index=True)
        else:
            st.error(f"Estado: {r1['estado']}")
