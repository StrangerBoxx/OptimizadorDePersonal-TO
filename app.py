"""
Versión web ULTRA SIMPLE del modelo (mismo modelo que Modelo_Simple.py).

Ejecutar: doble clic en Abrir_App.bat
      o:  python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import pulp
import streamlit as st

st.set_page_config(page_title="Dotación SCEL (simple)", layout="wide")
st.title("¿Cuánto personal necesita el aeropuerto?")

# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
@st.cache_data
def cargar():
    datos = pd.read_csv(Path(__file__).parent / "datos" / "Aeropuerto_SCEL_-_Hoja_1.csv")
    datos["Fecha"] = pd.to_datetime(datos["Fecha"], dayfirst=True)
    return datos

datos = cargar()
HORAS = range(24)

# ---------------------------------------------------------------------------
# 2. SUPUESTOS (barra lateral)
# ---------------------------------------------------------------------------
st.sidebar.header("Supuestos")
DIA = st.sidebar.date_input("Día a planificar", pd.Timestamp("2025-09-02"),
                            min_value=datos["Fecha"].min(), max_value=datos["Fecha"].max())
ALFA = st.sidebar.slider("Mínimo de pasajeros atendidos (α)", 0.5, 1.0, 0.90, 0.01)
MAX_PERSONAL = st.sidebar.number_input("Máximo de personal en el día", 10, 2000, 360, 10)
MAX_PUESTOS = st.sidebar.number_input("Máximo de puestos por área", 1, 200, 40)
MULTA = st.sidebar.number_input("Multa por pasajero no atendido (CLP)", 0, 100000, 1500, 100)

# Cada área: (ruta, % que pasa por ahí, horas antes del vuelo que llega, pax que atiende 1 persona por hora)
AREAS = {
    "Checkin_NAC":   ("NACIONAL",      0.35, 2, 25),
    "Seguridad_NAC": ("NACIONAL",      1.00, 1, 60),
    "Embarque_NAC":  ("NACIONAL",      1.00, 0, 80),
    "Checkin_INT":   ("INTERNACIONAL", 0.70, 3, 25),
    "Seguridad_INT": ("INTERNACIONAL", 1.00, 2, 60),
    "Embarque_INT":  ("INTERNACIONAL", 1.00, 0, 80),
}
TURNOS = {"FT": (8, 8000), "PT": (4, 9000)}  # tipo: (duración en horas, costo por hora)

with st.sidebar.expander("Ver supuestos fijos"):
    st.dataframe(pd.DataFrame(AREAS, index=["Ruta", "% que pasa", "Anticipo (h)", "Pax/persona/h"]).T)
    st.write("Turnos: 8 h a $8.000/h (FT) y 4 h a $9.000/h (PT)")

# Pasajeros que despegan en cada hora
dia = datos[datos["Fecha"] == pd.Timestamp(DIA)]
pax = {(r, h): dia[(dia["Ruta"] == r) & (dia["Bloque Horario"] == h)]["Pax Total Bloque"].sum()
       for r in ["NACIONAL", "INTERNACIONAL"] for h in HORAS}

# Demanda por área y hora
D = {(a, h): uso * pax[ruta, (h + anticipo) % 24]
     for a, (ruta, uso, anticipo, _) in AREAS.items() for h in HORAS}

def horas_turno(tipo, inicio):
    return [(inicio + k) % 24 for k in range(TURNOS[tipo][0])]

LISTA_TURNOS = [(tipo, ini) for tipo in TURNOS for ini in HORAS]

# ---------------------------------------------------------------------------
# 3. MODELO
# ---------------------------------------------------------------------------
m = pulp.LpProblem("Dotacion", pulp.LpMinimize)
z = pulp.LpVariable.dicts("z", [(a, s) for a in AREAS for s in LISTA_TURNOS], lowBound=0, cat="Integer")
d = pulp.LpVariable.dicts("d", [(a, h) for a in AREAS for h in HORAS], lowBound=0)

def x(a, h):
    return pulp.lpSum(z[a, s] for s in LISTA_TURNOS if h in horas_turno(*s))

m += (pulp.lpSum(TURNOS[s[0]][0] * TURNOS[s[0]][1] * z[a, s] for a in AREAS for s in LISTA_TURNOS)
      + MULTA * pulp.lpSum(d.values()))
for a, (_, _, _, prod) in AREAS.items():
    for h in HORAS:
        m += prod * x(a, h) + d[a, h] >= D[a, h]
        m += d[a, h] <= (1 - ALFA) * D[a, h]
        m += x(a, h) <= MAX_PUESTOS
m += pulp.lpSum(z.values()) <= MAX_PERSONAL

with st.spinner("Resolviendo…"):
    m.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

# ---------------------------------------------------------------------------
# 4. RESULTADOS
# ---------------------------------------------------------------------------
estado = pulp.LpStatus[m.status]
if estado != "Optimal":
    st.error(f"No hay solución ({estado}). Prueba subir el máximo de personal o de puestos, o bajar α.")
    st.stop()

personal = int(round(sum(v.value() for v in z.values())))
no_atendidos = sum(v.value() for v in d.values())
total_dem = sum(D.values())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pasajeros del día", f"{sum(pax.values()):,.0f}".replace(",", "."))
c2.metric("Costo total", f"${pulp.value(m.objective):,.0f}".replace(",", "."))
c3.metric("Personal contratado", f"{personal} / {MAX_PERSONAL}")
c4.metric("No atendidos", f"{100 * no_atendidos / total_dem:.1f} %")

st.subheader("Personas trabajando por hora")
tabla = pd.DataFrame({a: [int(round(pulp.value(x(a, h)))) for h in HORAS] for a in AREAS}).T
tabla.columns = [f"{h}h" for h in HORAS]
st.dataframe(tabla.style.background_gradient(cmap="Blues", axis=None), width="stretch")

st.subheader("Demanda vs. capacidad por área")
area = st.selectbox("Área", list(AREAS))
prod = AREAS[area][3]
graf = pd.DataFrame({"Pasajeros que llegan": [D[area, h] for h in HORAS],
                     "Pasajeros que se pueden atender": [prod * tabla.loc[area].iloc[h] for h in HORAS]},
                    index=HORAS)
st.line_chart(graf)

st.subheader("Turnos elegidos")
turnos = [{"Área": a, "Tipo": s[0], "Inicio": f"{s[1]}:00", "Fin": f"{(s[1] + TURNOS[s[0]][0]) % 24}:00",
           "Personas": int(round(z[a, s].value()))}
          for a in AREAS for s in LISTA_TURNOS if z[a, s].value() > 0.5]
st.dataframe(pd.DataFrame(turnos), hide_index=True, width="stretch")
