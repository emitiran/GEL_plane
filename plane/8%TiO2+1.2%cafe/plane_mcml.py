import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import subprocess

import matplotlib.pyplot as plt



import pandas as pd

import matplotlib.pyplot as plt

def write_best_estimate(best_estimate, x1, x2, x3, x4, d, phantom_mci):
    with open("best_estimate.txt", "w") as f:
        f.write("The overall best estimate is: " + str(best_estimate) + "\n")

        mua, mus, g = best_estimate
        Rcd, Tcd, Abs, Tc = black_box((mua, mus, g, d), phantom_mci.copy())

        f.write(f"Rcd_o: {x1:.4f}, Tcd_o: {x2:.4f}, Tc_o: {x4:.4f}, A_o: {x3:.4f}\n")
        f.write(f"Rcd  : {Rcd:.4f}, Tcd  : {Tcd:.4f}, Tc  : {Tc:.4f}, A  : {Abs:.4f}\n")
        f.write(f"mua: {mua:.4f}, mus: {mus:.4f}, g: {g:.4f}\n")

        relative_error = objective_function(best_estimate, x1, x2, x3, x4, d, phantom_mci.copy())
        f.write(f"Error relativo: {relative_error[0]:.4f}\n")
        a = mus/(mua+mus)
        tau = d*(mua+mus)
        f.write(f"a: {a:.4f}, tau: {tau:.4f}, g: {g:.4f}\n")

def plot_values2(data1, data2, Rcd_meas, Tcd_meas, Rcd, Tcd):
    plt.figure(figsize=(10, 6))

    datasets = [(data1, 'blue', 'Data 1'), (data2, 'red', 'Data 2')]
    all_data = []
    for data, color, label in datasets:
        a_values, g_values, R_values, T_values = data  # Unpack the a, g, R, T values
        plt.scatter(R_values, T_values, color=color, label=label)

        for a, g, R, T in zip(a_values, g_values, R_values, T_values):
            plt.text(R, T, f'({a:.2f}, {g:.2f})', fontsize=8, ha='right')
        # Add data to the total data
        all_data.extend(zip(a_values, g_values, R_values, T_values))

    # Save all data to csv file
    df = pd.DataFrame(all_data, columns=['a', 'g', 'R', 'T'])
    df.to_csv('all_data.csv', index=False)


    # Plot Rcd_meas, Tcd_meas with a black cross
    plt.scatter(Rcd_meas, Tcd_meas, color='black', marker='x', s=100, label='Measured Rcd, Tcd')

    # Plot Rcd, Tcd with a red circle
    plt.scatter(Rcd, Tcd, edgecolors='red', facecolors='none', marker='o', s=100, label='Rcd, Tcd')


    plt.xlabel('R values')
    plt.ylabel('T values')
    plt.title('R vs T plot for different a, g values')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()



def generate_values(n,ai, af,gi, gf,  phantom_mci, tau, d):
    # Genera n puntos aleatorios para a y g
    a_values = np.random.uniform(ai, af, n)
    g_values = np.random.uniform(gi, gf, n)

    # Listas para guardar los valores de salida
    a_output = []
    g_output = []
    Rcd_output = []
    Tcd_output = []

    # Calcula mua, mus, Rcd y Tcd para cada par de valores a y g
    for a, g in zip(a_values, g_values):
        mua = tau*(1 - a)/d
        mus = a*tau/d

        # Usar estos valores como entrada para la función black_box
        Rcd, Tcd, Abs, Tc = black_box((mua, mus, g, d), phantom_mci)

        # Guardar los valores de a, g, Rcd y Tcd
        a_output.append(a)
        g_output.append(g)
        Rcd_output.append(Rcd)
        Tcd_output.append(Tcd)

    # Crea un DataFrame con los valores de salida
    output_values = pd.DataFrame({
        'a': a_output,
        'a2': a_output,
        'a3': a_output,
        'a4': a_output,
        'g': g_output,
        'a5': a_output,
        'Rcd': Rcd_output,
        'Tcd': Tcd_output
    })

    return output_values



            
def compute_quadratic_coefficients(df1, df2):
    # Extraer los datos de las columnas específicas
    x = np.concatenate([df1.iloc[:,0], df2.iloc[:,0]])  # Primer columna corresponde a "a"
    y = np.concatenate([df1.iloc[:,4], df2.iloc[:,4]])  # Quinta columna corresponde a "g"
    R = np.concatenate([df1.iloc[:,6], df2.iloc[:,6]])  # Séptima columna corresponde a "Rcd"
    T = np.concatenate([df1.iloc[:,7], df2.iloc[:,7]])  # Octava columna corresponde a "Tcd"

    # Construir las matrices de coeficientes para el ajuste de segundo orden
    A_R = np.vstack([x**2, y**2, x*y, x, y, np.ones(len(x))]).T
    A_T = np.vstack([x**2, y**2, x*y, x, y, np.ones(len(x))]).T

    # Resuelve el sistema de ecuaciones para R
    params_R = np.linalg.lstsq(A_R, R, rcond=None)[0]

    # Resuelve el sistema de ecuaciones para T
    params_T = np.linalg.lstsq(A_T, T, rcond=None)[0]

    # Los coeficientes del modelo
    A, B, C, D, E, F = params_R  # Coeficientes para R
    G, H, I, J, K, L = params_T  # Coeficientes para T

    return A, B, C, D, E, F, G, H, I, J, K, L



def compute_coefficients(df1, df2):
    # Extraer los datos de las columnas específicas
    x = np.concatenate([df1.iloc[:,0], df2.iloc[:,0]]) # Primer columna corresponde a "a"
    y = np.concatenate([df1.iloc[:,4], df2.iloc[:,4]]) # Quinta columna corresponde a "g"
    R = np.concatenate([df1.iloc[:,6], df2.iloc[:,6]]) # Séptima columna corresponde a "Rcd"
    T = np.concatenate([df1.iloc[:,7], df2.iloc[:,7]]) # Octava columna corresponde a "Tcd"

    # Construye la matriz de coeficientes
    A_R = np.vstack([x, y, np.ones(len(x))]).T
    A_T = np.vstack([x, y, np.ones(len(x))]).T

    # Resuelve el sistema de ecuaciones para R
    params_R = np.linalg.lstsq(A_R, R, rcond=None)[0]

    # Resuelve el sistema de ecuaciones para T
    params_T = np.linalg.lstsq(A_T, T, rcond=None)[0]

    # Los coeficientes del modelo
    A, B, C = params_R  # Coeficientes para R
    D, E, F = params_T  # Coeficientes para T

    return A, B, C, D, E, F


def plot_values(data1, data2, Rcd_meas, Tcd_meas, Rcd, Tcd):
    plt.figure(figsize=(10, 6))

    datasets = [(data1, 'blue', 'Data 1'), (data2, 'red', 'Data 2')]
    for data, color, label in datasets:
        a_values, R_values, T_values = data  # Descomprime los valores de a, R, T
        plt.scatter(R_values, T_values, color=color, label=label)

        for a, R, T in zip(a_values, R_values, T_values):
            plt.text(R, T, str(a), fontsize=12, ha='right')

    # Plot Rcd_meas, Tcd_meas with a black cross
    plt.scatter(Rcd_meas, Tcd_meas, color='black', marker='x', s=100, label='Measured Rcd, Tcd')
    
     # Plot Rcd, Tcd with a red circle
    # Marcador círculo vacío
    plt.scatter(Rcd, Tcd, edgecolors='red', facecolors='none', marker='o', s=100, label='Rcd, Tcd')

    plt.xlabel('R values')
    plt.ylabel('T values')
    plt.title('R vs T plot for different a values')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def objective_function(individual, x1, x2, x3, x4, d, phantom_mci):
    mua, mus, g = individual  # added "g"
    Rcd, Tcd, Abs, Tc = black_box((mua, mus, g, d), phantom_mci.copy())  # included "g" in the function call

    print("Rcd_o: {:.4f}, Tcd_o: {:.4f}, Tc_o: {:.4f}, A_o: {:.4f}".format(x1, x2, x4, x3))
    print("Rcd  : {:.4f}, Tcd  : {:.4f}, Tc  : {:.4f}, A  : {:.4f}".format(Rcd, Tcd, Tc, Abs))
    print("mua: {:.4f}, mus: {:.4f}, g: {:.4f}".format(mua, mus, g))

    out_val1 = np.abs(x1 - Rcd)/(Rcd+1e-9)
    out_val2 = np.abs(x2 - Tcd)/(Tcd + 1e-9)
    out_val3 = np.abs(x4 - Tc)/(Tc + 1e-9)
    
    if x4 <= 0.01 and Tc <= 0.01:
        out_val = out_val1 + out_val2
    else:
        out_val = out_val1 + out_val2 + out_val3 

    print("Error relativo: {:.4f}".format(out_val))
    print("______________________________")  # added line

    return out_val,  # Notice the comma at the end, which makes this a tuple

def read_input_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove any leading or trailing spaces and split by "#"
    lines = [line.strip().split('#') for line in lines]

    # Only consider values before "#", if it exists
    lines = [[value.strip() for value in line[0].split(",")] if line else [] for line in lines]

    # Remove empty lines
    lines = [line for line in lines if line]
    #print(lines) 
    n_values = [float(value) for value in lines[0]]
    g_values = [(value) for value in lines[1]]
    a_values = [(value) for value in lines[2]]
    tau_values = [float(value) for value in lines[3]]
    d = float(lines[4][0])  # Assuming d is a single value on its line
    Rcd, Tcd, Tc = lines[5]

    #print("",Rcd, Tcd, Tc)

    return n_values, g_values, a_values, tau_values, d, float(Rcd), float(Tcd), float(Tc)

# Define la función black_box
import tempfile
def black_box(inputs, phantom_mci):
    """
    Función que realiza los cálculos y genera los valores de salida (Rcd, Tcd, Abs, Tc) a partir de los parámetros de entrada.

    Args:
        inputs: Tupla que contiene los parámetros de entrada.
        phantom_mci: Contenido del archivo phantom.mci a modificar.

    Returns:
        tuple: Tupla con los valores de salida (Rcd, Tcd, Abs, Tc).
    """
    mua, mus, g, d = inputs

    # Read the values from line 19
    t1, _, _, _, t5, *_ = phantom_mci[20].split()
    t5 = d
    #print(t1)
    # Write back the values to line 19, replacing t2, t3, and t4
    phantom_mci[20] = "{}\t{}\t{}\t{}\t{}\n".format(t1, mua, mus, g, t5)

    # Use a temporary file instead of "phantom2.mci"
    with tempfile.NamedTemporaryFile(suffix=".mci", delete=True, mode="w+") as temp:
        temp.writelines(phantom_mci)
        temp.flush()  # Ensure the data is written to the file
        #print(phantom_mci)
        #print(temp.flush())
        # Reset file cursor to the beginning
        temp.seek(0)
        
        # Read and print file contents for verification
        #print("Temporary file contents:")
        #print(temp.read())

        try:
            result = subprocess.run(["./mcml", temp.name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            output_lines = result.stdout.decode().split('\n')  # Split stdout into lines
            #print(output_lines[33])
            Rcd_line = output_lines[33]
            Abs_line = output_lines[34]
            Tcd_line = output_lines[36]
            Tc_line = output_lines[37]

            Rcd = float(Rcd_line.split()[0])
            Tcd = float(Tcd_line.split()[0])
            Abs = float(Abs_line.split()[0])
            Tc = float(Tc_line.split()[0])
        except subprocess.CalledProcessError:
            return float('inf'), float('inf'), float('inf'), float('inf')
    #print("Black_box; Rcd  : {:.4f}, Tcd  : {:.4f}, Tc  : {:.4f}, A  : {:.4f}".format(Rcd, Tcd, Tc, Abs))
    #print("Black_box; mua: {:.4f}, mus: {:.4f}, g: {:.4f}".format(mua, mus, g))
    return Rcd, Tcd, Abs, Tc

import random
import time
def main():
    random.seed(time.time())
    
    try:
        with open("phantom.mci", "r") as file:
                phantom_mci = file.readlines()
    except IOError:
        print("Error: unable to open phantom.mci for reading")
        return

    n, g_values, a_values, tau_values, d, Rcd_meas, Tcd_meas, Tc_meas = read_input_file("input_plane.dat")
    tau = float(tau_values[0])
    print(f"Rcd = {Rcd_meas}")
    print(f"Tcd = {Tcd_meas}")
    print(f"tau = {tau}")

    

    # Leer los datos del archivo CSV
    #df1 = pd.read_csv("results1.csv")
    #df2 = pd.read_csv("results2.csv")
    n_divided_by_2_rounded = [round(i / 2) for i in n]

    #print(g_values)
    #ai, af = map(float, a_values[0].split())
    ai, af = map(float, a_values)
    gi, gf = map(float, g_values)



    df1 = generate_values(n_divided_by_2_rounded,ai, af, gi, gf, phantom_mci, tau, d)
    df2 = generate_values(n_divided_by_2_rounded,ai, af, gi, gf, phantom_mci, tau, d)

    
    model_o = 1
    if model_o == 1:
        A, B, C, D, E, F = compute_coefficients(df1, df2)

        pr1 = 0
        if pr1 == 1:
            print("A = ", A)
            print("B = ", B)
            print("C = ", C)
            print("D = ", D)
            print("E = ", E)
            print("F = ", F)

        # Matriz de coeficientes
        A_matrix = np.array([[A, B], [D, E]])
        # Vector constante
        B_vector = np.array([Rcd_meas - C, Tcd_meas - F])

        # Matrices A_x y A_y
        A_x = A_matrix.copy()
        A_x[:, 0] = B_vector

        A_y = A_matrix.copy()
        A_y[:, 1] = B_vector

        # Calcular los determinantes
        det_A = np.linalg.det(A_matrix)
        det_A_x = np.linalg.det(A_x)
        det_A_y = np.linalg.det(A_y)

        # Calcular x y y
        xo = det_A_x / det_A
        yo = det_A_y / det_A

    if model_o == 1:
        A, B, C, D, E, F, G, H, I, J, K, L = compute_quadratic_coefficients(df1, df2)
        pr1 = 0
        if pr1 == 1:
            print("A = ", A)
            print("B = ", B)
            print("C = ", C)
            print("D = ", D)
            print("E = ", E)
            print("F = ", F)
            print("G = ", G)
            print("H = ", H)
            print("I = ", I)
            print("J = ", J)
            print("K = ", K)
            print("L = ", L)


        # Los valores medidos de Rcd y Tcd
        Rcd, Tcd = Rcd_meas, Tcd_meas

        # Define las ecuaciones
        def equations(vars):
            x, y = vars
            eq1 = A*x**2 + B*y**2 + C*x*y + D*x + E*y + F - Rcd
            eq2 = G*x**2 + H*y**2 + I*x*y + J*x + K*y + L - Tcd
            return [eq1, eq2]


        # Adivinación inicial de las soluciones
        x_guess, y_guess = xo, yo
        solution = fsolve(equations, (x_guess, y_guess))

        x = solution[0]
        y = solution[1]

        print(f'Las soluciones son x = {xo} e y = {yo}')
        print(f'Las soluciones son x = {solution[0]} e y = {solution[1]}')



    try:
        with open("phantom.mci", "r") as file:
            phantom_mci = file.readlines()
    except IOError:
        print("Error: unable to open phantom.mci for reading")
        return


    print(f"a = {x}")
    print(f"g = {y}")

    a = x
    g = np.abs(y)
   
    mua = tau*(1 - a)/d
    mus = a*tau/d
    inputs = mua, mus, g, d
    Rcd, Tcd, Abs, Tc = black_box(inputs, phantom_mci)
    
    individual = mua, mus, g
    x1, x2, x3, x4 = Rcd_meas, Tcd_meas, 1-(Rcd_meas+Tcd_meas +Tc_meas ), Tc_meas 
    objective_function(individual, x1, x2, x3, x4, d, phantom_mci)



    best_estimate =   mua, mus,g
    write_best_estimate(best_estimate, x1, x2, x3, x4, d, phantom_mci)

    #data1 = df1.iloc[:,0], df1.iloc[:,4], df1.iloc[:,6], df1.iloc[:,7]
    #data2 = df2.iloc[:,0], df2.iloc[:,4], df2.iloc[:,6], df2.iloc[:,7]
    #plot_values2(data1, data2, Rcd_meas, Tcd_meas, Rcd, Tcd)

    

if __name__ == '__main__':
    main()