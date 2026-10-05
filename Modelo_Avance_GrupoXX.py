"""
ELI148 - Avance de Proyecto
Asignación de personal de atención en el Aeropuerto Arturo Merino Benítez (SCEL)
según demanda horaria real de pasajeros. Modelo MILP base.

Requiere : pip install "pulp==2.9.0" pandas
Ejecutar : python Modelo_Avance_GrupoXX.py
Datos    : carpeta ./datos con los CSV descargados de la JAC
  - Aeropuerto_SCEL_-_Hoja_1.csv  (Pasajeros embarcados por hora, por día, nac./intl.)
  - operaciones-aeropuertos.csv    (Operaciones de aeronaves mensuales por aeropuerto)
  - trafico.csv                    (Tráfico aéreo mensual por operador y ruta)

DATO REAL  : pasajeros embarcados por hora y ruta; operaciones mensuales.
SUPUESTOS  : llegada anticipada, uso de mesón, productividad, costos, capacidades,
             penalización, nivel de servicio y personal disponible (marcados abajo).
"""
import math
import time
from pathlib import Path

import pandas as pd
import pulp

DATOS = Path(__file__).parent / "datos"

# ---------------------------------------------------------------------------
# 1. LECTURA DE DATOS REALES (JAC)
# ---------------------------------------------------------------------------
def cargar_embarcados():
    d = pd.read_csv(DATOS / "Aeropuerto_SCEL_-_Hoja_1.csv")
    d["Fecha"] = pd.to_datetime(d["Fecha"], dayfirst=True)
    return d.rename(columns={"Bloque Horario": "hora", "Pax Total Bloque": "pax",
                             "Asientos Ofrecidos Bloque": "asientos", "Ruta": "ruta"})


def salidas_del_dia(emb, fecha):
    """Pasajeros embarcados por hora (0-23) en SCEL, separados en NAC e INT."""
    x = emb[emb["Fecha"] == pd.Timestamp(fecha)]
    tab = x.pivot_table(index="hora", columns="ruta", values="pax", aggfunc="sum")
    tab = tab.reindex(range(24)).fillna(0)
    return {"NAC": tab.get("NACIONAL", pd.Series(0, index=range(24))).to_dict(),
            "INT": tab.get("INTERNACIONAL", pd.Series(0, index=range(24))).to_dict()}


def pax_por_salida(emb, anio_mes):
    """Pasajeros promedio por despegue = pax embarcados / (operaciones / 2).
    Supuesto: la mitad de las operaciones del mes son despegues."""
    ops = pd.read_csv(DATOS / "operaciones-aeropuertos.csv")
    ops = ops[(ops.aeropuerto_oaci == "SCEL") & (ops.mes_id == anio_mes)]
    ops = ops.set_index("internacional_domestico")["cnt_operaciones"]
    a, m = divmod(anio_mes, 100)
    e = emb[(emb["Fecha"].dt.year == a) & (emb["Fecha"].dt.month == m)]
    pax = e.groupby("ruta")["pax"].sum()
    return {"NAC": pax["NACIONAL"] / (ops["D"] / 2), "INT": pax["INTERNACIONAL"] / (ops["I"] / 2)}


def validar_con_trafico(emb, anio, mes):
    """Contrasta el total mensual del archivo horario con el de Tráfico Aéreo Mensual."""
    t = pd.read_csv(DATOS / "trafico.csv", sep=";")
    tot_t = t[(t["ORIG_1"] == "SCL") & (t["Año"] == anio) & (t["Mes"] == mes)]["PASAJEROS"].sum()
    e = emb[(emb["Fecha"].dt.year == anio) & (emb["Fecha"].dt.month == mes)]["pax"].sum()
    return e, tot_t, (e - tot_t) / tot_t


# ---------------------------------------------------------------------------
# 2. PARÁMETROS (SUPUESTOS declarados)
# ---------------------------------------------------------------------------
AREAS = ["Checkin_NAC", "Seguridad_NAC", "Embarque_NAC",
         "Checkin_INT", "Seguridad_INT", "Embarque_INT"]
USO_MESON = {"NAC": 0.35, "INT": 0.70}          # SUPUESTO: resto hace check-in web
# SUPUESTO llegada anticipada: {horas antes del vuelo: fracción de pasajeros}
ANTICIPO = {
    ("Checkin", "NAC"): {2: 0.6, 1: 0.4},
    ("Checkin", "INT"): {3: 0.5, 2: 0.5},
    ("Seguridad", "NAC"): {1: 0.6, 0: 0.4},
    ("Seguridad", "INT"): {2: 0.5, 1: 0.5},
    ("Embarque", "NAC"): {0: 1.0},
    ("Embarque", "INT"): {0: 1.0},
}
PROD_BASE = {"Checkin": 25, "Seguridad": 60}     # SUPUESTO pax/(trabajador*h)
AGENTES_POR_VUELO = 2                            # SUPUESTO para embarque
CAPACIDAD = {"Checkin_NAC": 40, "Seguridad_NAC": 36, "Embarque_NAC": 40,
             "Checkin_INT": 50, "Seguridad_INT": 30, "Embarque_INT": 40}  # SUPUESTO puestos
COSTO_HORA = {"FT": 8000, "PT": 9000}            # SUPUESTO CLP/h
PENALIZACION = 1500                              # SUPUESTO CLP por pasajero no atendido
ALFA = 0.90                                      # SUPUESTO nivel mínimo de servicio
W_MAX = 360                                      # SUPUESTO personal disponible por día


def demanda_por_area(salidas, horizonte, factor=1.0):
    """Demanda (pax/h) por área. El horizonte es cíclico de 24 h (operación continua)."""
    D = {}
    for a in AREAS:
        tipo, ruta = a.split("_")
        uso = USO_MESON[ruta] if tipo == "Checkin" else 1.0
        D[a] = {t: factor * uso * sum(f * salidas[ruta][(t + k) % 24]
                                     for k, f in ANTICIPO[tipo, ruta].items())
                for t in horizonte}
    return D


def productividad(pps, factor=1.0):
    mu = {}
    for a in AREAS:
        tipo, ruta = a.split("_")
        base = PROD_BASE.get(tipo, pps[ruta] / AGENTES_POR_VUELO)  # embarque: pax/agente/h
        mu[a] = base * factor
    return mu


def patrones_turno(n_horas, largo_ft=8, largo_pt=4, ciclico=True):
    """Turnos de 8 h y 4 h que pueden iniciar en cualquier hora; cruzan medianoche."""
    pats = {}
    for tipo, L in (("FT", largo_ft), ("PT", largo_pt)):
        for ini in range(n_horas):
            if not ciclico and ini + L > n_horas:
                continue
            pats[f"{tipo}{ini:03d}"] = {"horas": {(ini + k) % n_horas for k in range(L)},
                                       "costo": COSTO_HORA[tipo] * L}
    return pats


# ---------------------------------------------------------------------------
# 3. MODELO MILP
# ---------------------------------------------------------------------------
def resolver(areas, horas, D, mu, K, pats, W, alfa=ALFA, P=PENALIZACION, tiempo_max=120):
    m = pulp.LpProblem("Dotacion_SCEL", pulp.LpMinimize)
    S = list(pats)
    z = pulp.LpVariable.dicts("z", (areas, S), lowBound=0, cat="Integer")
    x = pulp.LpVariable.dicts("x", (areas, horas), lowBound=0, cat="Integer")
    d = pulp.LpVariable.dicts("d", (areas, horas), lowBound=0)

    m += (pulp.lpSum(pats[s]["costo"] * z[a][s] for a in areas for s in S)
          + P * pulp.lpSum(d[a][t] for a in areas for t in horas))
    for a in areas:
        for t in horas:
            m += x[a][t] == pulp.lpSum(z[a][s] for s in S if t in pats[s]["horas"])  # R1
            m += mu[a] * x[a][t] + d[a][t] >= D[a][t]                                # R2
            m += d[a][t] <= (1 - alfa) * D[a][t]                                     # R3
            m += x[a][t] <= K[a]                                                     # R5
    m += pulp.lpSum(z[a][s] for a in areas for s in S) <= W                          # R4

    solver = (pulp.HiGHS(msg=False, timeLimit=tiempo_max) if "HiGHS" in pulp.listSolvers(True)
              else pulp.PULP_CBC_CMD(msg=False, timeLimit=tiempo_max))
    t0 = time.perf_counter()
    m.solve(solver)
    res = {"estado": pulp.LpStatus[m.status], "tiempo_s": time.perf_counter() - t0,
           "n_var": m.numVariables(), "n_res": m.numConstraints()}
    if res["estado"] != "Optimal":
        return res
    res["objetivo"] = pulp.value(m.objective)
    res["z"] = {(a, s): int(round(z[a][s].value())) for a in areas for s in S if z[a][s].value() > 0.5}
    res["x"] = {(a, t): int(round(x[a][t].value())) for a in areas for t in horas}
    res["d"] = {(a, t): d[a][t].value() for a in areas for t in horas}
    res["trabajadores"] = sum(res["z"].values())
    res["deficit_total"] = sum(res["d"].values())
    res["demanda_total"] = sum(D[a][t] for a in areas for t in horas)
    return res


def verificar(res, areas, horas, D, mu, K, pats, W, alfa=ALFA, tol=1e-6):
    err = []
    for a in areas:
        for t in horas:
            xc = sum(n for (aa, s), n in res["z"].items() if aa == a and t in pats[s]["horas"])
            if xc != res["x"][a, t]: err.append(f"R1 {a} {t}")
            if mu[a] * res["x"][a, t] + res["d"][a, t] < D[a][t] - tol: err.append(f"R2 {a} {t}")
            if res["d"][a, t] > (1 - alfa) * D[a][t] + tol: err.append(f"R3 {a} {t}")
            if res["x"][a, t] > K[a]: err.append(f"R5 {a} {t}")
    if res["trabajadores"] > W: err.append("R4")
    return err or "OK"


def resumen(nombre, r):
    if r["estado"] != "Optimal":
        print(f"{nombre:34s} {r['estado']}")
        return
    print(f"{nombre:34s} obj=${r['objetivo']:>12,.0f}  personal={r['trabajadores']:4d}  "
          f"no atendidos={r['deficit_total']:7.1f} pax ({100*r['deficit_total']/r['demanda_total']:.2f} %)  "
          f"t={r['tiempo_s']:.2f} s")


# ---------------------------------------------------------------------------
# 4. EXPERIMENTOS
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    emb = cargar_embarcados()
    e, t, err = validar_con_trafico(emb, 2025, 1)
    print(f"Validación ene-2025: horario={e:,.0f} pax | tráfico mensual={t:,.0f} pax | diferencia={100*err:+.2f} %")
    pps = pax_por_salida(emb, 202509)
    print(f"Pax por despegue (sep-2025): NAC={pps['NAC']:.0f}  INT={pps['INT']:.0f}")
    print(f"Factor de ocupación 2025: {emb[emb.Fecha.dt.year==2025].pax.sum()/emb[emb.Fecha.dt.year==2025].asientos.sum():.1%}")

    # --- Instancia 1: verificación (1 área, 6 horas, 3 turnos) ---------------
    H1 = list(range(6))
    D1 = {"Checkin_NAC": dict(zip(H1, [100, 150, 200, 150, 100, 50]))}
    pats1 = {"T06": {"horas": {0, 1, 2, 3}, "costo": 36000},
             "T08": {"horas": {2, 3, 4, 5}, "costo": 36000},
             "T06L": {"horas": {0, 1, 2, 3, 4, 5}, "costo": 50000}}
    r1 = resolver(["Checkin_NAC"], H1, D1, {"Checkin_NAC": 25}, CAPACIDAD, pats1, W_MAX)
    print("\n=== Instancia 1 - verificación ===")
    resumen("1 área, 6 h, 3 turnos", r1)
    print("Turnos:", r1["z"], "| Verificación:",
          verificar(r1, ["Checkin_NAC"], H1, D1, {"Checkin_NAC": 25}, CAPACIDAD, pats1, W_MAX))

    # --- Instancia 2: día típico real (6 áreas, 24 h, 48 patrones) ------------
    H = list(range(24))
    pats = patrones_turno(24)
    mu = productividad(pps)
    DIA_TIPICO = "2025-09-02"
    D2 = demanda_por_area(salidas_del_dia(emb, DIA_TIPICO), H)
    r2 = resolver(AREAS, H, D2, mu, CAPACIDAD, pats, W_MAX)
    print(f"\n=== Instancia 2 - día típico real ({DIA_TIPICO}) ===")
    resumen("6 áreas, 24 h", r2)
    print("Verificación:", verificar(r2, AREAS, H, D2, mu, CAPACIDAD, pats, W_MAX))
    print(f"Variables={r2['n_var']}  Restricciones={r2['n_res']}")
    for a in AREAS:
        print(f"  {a:14s}", " ".join(f"{r2['x'][a, t]:2d}" for t in H))

    # --- Escenarios ----------------------------------------------------------
    print("\n=== Escenarios (días reales) ===")
    escenarios = {
        "Típico  mar 02-09-2025": ("2025-09-02", 1.0),
        "Peak    lun 02-02-2026": ("2026-02-02", 1.0),
        "Bajo    sáb 23-05-2026": ("2026-05-23", 1.0),
        "Típico, productividad -20 %": ("2025-09-02", 0.8),
    }
    for nom, (f, pf) in escenarios.items():
        sal = salidas_del_dia(emb, f)
        D = demanda_por_area(sal, H)
        mu_e = productividad(pps, pf)
        r = resolver(AREAS, H, D, mu_e, CAPACIDAD, pats, W_MAX)
        tot = sum(sal["NAC"].values()) + sum(sal["INT"].values())
        if r["estado"] == "Optimal":
            resumen(f"{nom} ({tot:,.0f} pax)", r)
        else:
            r_sin = resolver(AREAS, H, D, mu_e, CAPACIDAD, pats, 10**6)
            msg = (f"se necesitan {r_sin['trabajadores']} personas (obj=${r_sin['objetivo']:,.0f})"
                   if r_sin["estado"] == "Optimal" else "falta capacidad física (puestos) en: " +
                   ", ".join(a for a in AREAS if any((1 - ALFA) * D[a][t] < D[a][t] - mu_e[a] * CAPACIDAD[a] for t in H)))
            print(f"{nom} ({tot:,.0f} pax)  INFACTIBLE con W={W_MAX}: {msg}")

    # --- Escala: horizonte de varios días consecutivos -----------------------
    print("\n=== Escalabilidad (días reales consecutivos desde 01-09-2025) ===")
    for n_dias in (1, 3, 7, 14):
        fechas = pd.date_range("2025-09-01", periods=n_dias)
        Hn = list(range(24 * n_dias))
        Dn = {a: {} for a in AREAS}
        for i, f in enumerate(fechas):
            Dd = demanda_por_area(salidas_del_dia(emb, f), H)
            for a in AREAS:
                for t in H:
                    Dn[a][24 * i + t] = Dd[a][t]
        r = resolver(AREAS, Hn, Dn, mu, CAPACIDAD, patrones_turno(24 * n_dias), W_MAX * n_dias)
        print(f"{n_dias:2d} día(s): horas={len(Hn):4d}  variables={r['n_var']:6d}  "
              f"restricciones={r['n_res']:6d}  {r['estado']:10s}  t={r['tiempo_s']:6.2f} s")
