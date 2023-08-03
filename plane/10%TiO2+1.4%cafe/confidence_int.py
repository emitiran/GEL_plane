import os
import numpy as np
import scipy.stats as stats

def read_values():
    mua_values = []
    mus_values = []
    g_values = []
    Rcd_values = []
    Tcd_values = []
    Tc_values = []

    # Recorre todos los archivos en el directorio actual
    for filename in os.listdir("."):
        # Verifica si el nombre del archivo coincide con el patrón "best_estimate_run_*.txt"
        if filename.startswith("best_estimate_run_") and filename.endswith(".txt"):
            with open(filename, 'r') as file:
                lines = file.readlines()

                # Extrae los valores de mua, mus, g de la línea 4
                mua, mus, g = [float(val.split(": ")[1]) for val in lines[3].split(", ")]

                # Extrae los valores de Rcd, Tcd, Tc de la línea 3
                Rcd, Tcd, Tc,A = [float(val.split(": ")[1]) for val in lines[2].split(", ")]

                # Extrae el valor de dif de la línea 5
                dif = float(lines[4].split(": ")[1])

                # Filtra los valores donde dif es mayor que .1
                if dif <= .04:
                    mua_values.append(mua)
                    mus_values.append(mus)
                    g_values.append(g)
                    Rcd_values.append(Rcd)
                    Tcd_values.append(Tcd)
                    Tc_values.append(Tc)

    # Retorna los valores en una matriz
    return np.array([mua_values, mus_values, g_values, Rcd_values, Tcd_values, Tc_values])

values = read_values()

# Extraer los valores
mua_values = values[0]
mus_values = values[1]
g_values = values[2]
Rcd_values = values[3]
Tcd_values = values[4]
Tc_values = values[5]

# Prueba de Shapiro para verificar la normalidad
print("Prueba de Shapiro para mua:", stats.shapiro(mua_values), ", n = ", len(mua_values))
print("Prueba de Shapiro para mus:", stats.shapiro(mus_values), ", n = ", len(mus_values))
print("Prueba de Shapiro para g:", stats.shapiro(g_values), ", n = ", len(g_values))
print("Prueba de Shapiro para Rcd:", stats.shapiro(Rcd_values), ", n = ", len(Rcd_values))
print("Prueba de Shapiro para Tcd:", stats.shapiro(Tcd_values), ", n = ", len(Tcd_values))
print("Prueba de Shapiro para Tc:", stats.shapiro(Tc_values), ", n = ", len(Tc_values))

# Calcular el intervalo de confianza al 95%
ci_mua = stats.norm.interval(0.95, loc=np.mean(mua_values), scale=np.std(mua_values))
ci_mus = stats.norm.interval(0.95, loc=np.mean(mus_values), scale=np.std(mus_values))
ci_g = stats.norm.interval(0.95, loc=np.mean(g_values), scale=np.std(g_values))
ci_Rcd = stats.norm.interval(0.95, loc=np.mean(Rcd_values), scale=np.std(Rcd_values))
ci_Tcd = stats.norm.interval(0.95, loc=np.mean(Tcd_values), scale=np.std(Tcd_values))
ci_Tc = stats.norm.interval(0.95, loc=np.mean(Tc_values), scale=np.std(Tc_values))


# Imprimir los resultados en el formato deseado
print(f"mua: {np.mean(mua_values):.3f} (CI 95%: {ci_mua[0]:.3f}, {ci_mua[1]:.3f})")
print(f"mus: {np.mean(mus_values):.3f} (CI 95%: {ci_mus[0]:.3f}, {ci_mus[1]:.3f})")
print(f"g: {np.mean(g_values):.3f} (CI 95%: {ci_g[0]:.3f}, {ci_g[1]:.3f})")
print(f"Rcd: {np.mean(Rcd_values):.3f} (CI 95%: {ci_Rcd[0]:.3f}, {ci_Rcd[1]:.3f})")
print(f"Tcd: {np.mean(Tcd_values):.3f} (CI 95%: {ci_Tcd[0]:.3f}, {ci_Tcd[1]:.3f})")
print(f"Tc: {np.mean(Tc_values):.3f} (CI 95%: {ci_Tc[0]:.3f}, {ci_Tc[1]:.3f})")



# Abrir el archivo en modo escritura
with open('CI_output.txt', 'w') as file:
    # Imprimir los resultados en el formato deseado y escribirlos en el archivo
    print(f"mua: {np.mean(mua_values):.3f} (CI 95%: {ci_mua[0]:.3f}, {ci_mua[1]:.3f})", file=file)
    print(f"mus: {np.mean(mus_values):.3f} (CI 95%: {ci_mus[0]:.3f}, {ci_mus[1]:.3f})", file=file)
    print(f"g: {np.mean(g_values):.3f} (CI 95%: {ci_g[0]:.3f}, {ci_g[1]:.3f})", file=file)
    print(f"Rcd: {np.mean(Rcd_values):.3f} (CI 95%: {ci_Rcd[0]:.3f}, {ci_Rcd[1]:.3f})", file=file)
    print(f"Tcd: {np.mean(Tcd_values):.3f} (CI 95%: {ci_Tcd[0]:.3f}, {ci_Tcd[1]:.3f})", file=file)
    print(f"Tc: {np.mean(Tc_values):.3f} (CI 95%: {ci_Tc[0]:.3f}, {ci_Tc[1]:.3f})", file=file)


import pandas as pd

# Crear el DataFrame
df = pd.DataFrame({
    'mua': mua_values,
    'mus': mus_values,
    'g': g_values,
    'Rcd': Rcd_values,
    'Tcd': Tcd_values,
    'Tc': Tc_values
})

# Guardar el DataFrame en un archivo CSV
df.to_csv('dat_best-estimate.csv', index=False)

