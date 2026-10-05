"""
VERSIÓN ULTRA SIMPLE del modelo (para explicar).

Pregunta: ¿cuántos trabajadores contrato, y en qué turno, para atender
a los pasajeros de un día en el aeropuerto de Santiago (SCEL) al menor costo?

Ejecutar: python Modelo_Simple.py
"""
from pathlib import Path

import pandas as pd
import pulp

# ---------------------------------------------------------------------------
# 1. DATOS: pasajeros que salen en cada hora del día (dato real JAC)
# ---------------------------------------------------------------------------
DIA = "2025-09-02"
datos = pd.read_csv(Path(__file__).parent / "datos" / "Aeropuerto_SCEL_-_Hoja_1.csv")
datos["Fecha"] = pd.to_datetime(datos["Fecha"], dayfirst=True)
dia = datos[datos["Fecha"] == DIA]

HORAS = range(24)
pax = {}  # pax[(ruta, hora)] = pasajeros que despegan en esa hora
for ruta in ["NACIONAL", "INTERNACIONAL"]:
    for h in HORAS:
        filas = dia[(dia["Ruta"] == ruta) & (dia["Bloque Horario"] == h)]
        pax[ruta, h] = filas["Pax Total Bloque"].sum()

# ---------------------------------------------------------------------------
# 2. SUPUESTOS (inventados, se pueden cambiar)
# ---------------------------------------------------------------------------
# Cada área: (ruta, % de pasajeros que pasa por ahí, horas antes del vuelo que llega, pax que atiende 1 persona por hora)
AREAS = {
    "Checkin_NAC":   ("NACIONAL",      0.35, 2, 25),   # 65 % hace check-in web
    "Seguridad_NAC": ("NACIONAL",      1.00, 1, 60),
    "Embarque_NAC":  ("NACIONAL",      1.00, 0, 80),
    "Checkin_INT":   ("INTERNACIONAL", 0.70, 3, 25),   # 30 % hace check-in web
    "Seguridad_INT": ("INTERNACIONAL", 1.00, 2, 60),
    "Embarque_INT":  ("INTERNACIONAL", 1.00, 0, 80),
}
MAX_PUESTOS = 40      # máximo de personas trabajando a la vez en un área
MAX_PERSONAL = 360    # máximo de personas contratadas en el día
ALFA = 0.90           # hay que atender al menos el 90 % de la demanda
MULTA = 1500          # CLP por cada pasajero no atendido
TURNOS = {"FT": (8, 8000), "PT": (4, 9000)}  # tipo: (duración en horas, costo por hora)

# Demanda: pasajeros que llegan a un área en la hora h
# = (% que usa el área) x pasajeros cuyo vuelo sale "anticipo" horas después.
D = {(a, h): uso * pax[ruta, (h + anticipo) % 24]
     for a, (ruta, uso, anticipo, _) in AREAS.items() for h in HORAS}

# Turno = (tipo, hora de inicio). Ej: ("FT", 6) trabaja de 6 a 14.
def horas_turno(tipo, inicio):
    return [(inicio + k) % 24 for k in range(TURNOS[tipo][0])]

LISTA_TURNOS = [(tipo, ini) for tipo in TURNOS for ini in HORAS]

# ---------------------------------------------------------------------------
# 3. MODELO
# ---------------------------------------------------------------------------
m = pulp.LpProblem("Dotacion", pulp.LpMinimize)

# Variables
z = pulp.LpVariable.dicts("z", [(a, s) for a in AREAS for s in LISTA_TURNOS], lowBound=0, cat="Integer")  # personas por turno
d = pulp.LpVariable.dicts("d", [(a, h) for a in AREAS for h in HORAS], lowBound=0)                        # pax no atendidos

# Personas trabajando en el área a durante la hora h
def x(a, h):
    return pulp.lpSum(z[a, s] for s in LISTA_TURNOS if h in horas_turno(*s))

# Objetivo: costo de sueldos + multa por pasajeros no atendidos
m += (pulp.lpSum(TURNOS[s[0]][0] * TURNOS[s[0]][1] * z[a, s] for a in AREAS for s in LISTA_TURNOS)
      + MULTA * pulp.lpSum(d.values()))

# Restricciones
for a, (_, _, _, prod) in AREAS.items():
    for h in HORAS:
        m += prod * x(a, h) + d[a, h] >= D[a, h]     # atender la demanda (o dejarla sin atender)
        m += d[a, h] <= (1 - ALFA) * D[a, h]         # a lo más 10 % sin atender
        m += x(a, h) <= MAX_PUESTOS                  # no hay más puestos que estos
m += pulp.lpSum(z.values()) <= MAX_PERSONAL          # personal disponible

m.solve(pulp.PULP_CBC_CMD(msg=False))

# ---------------------------------------------------------------------------
# 4. RESULTADOS
# ---------------------------------------------------------------------------
print("Estado:", pulp.LpStatus[m.status])
print(f"Costo total: ${pulp.value(m.objective):,.0f}")
print("Personal contratado:", int(sum(v.value() for v in z.values())))
print("\nPersonas trabajando por hora (0 a 23):")
for a in AREAS:
    print(f"{a:14s}", " ".join(f"{int(round(pulp.value(x(a, h)))):2d}" for h in HORAS))
