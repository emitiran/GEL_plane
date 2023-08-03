import numpy as np
import subprocess
from deap import base, creator, tools, algorithms
import multiprocessing
import random

import tempfile
import subprocess

# Define la función black_box
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
    t1, _, _, _, t5, *_ = phantom_mci[19].split()
    t5 = d
    # Write back the values to line 19, replacing t2, t3, and t4
    phantom_mci[19] = "{}\t{}\t{}\t{}\t{}\n".format(t1, mua, mus, g, t5)

    # Use a temporary file instead of "phantom2.mci"
    with tempfile.NamedTemporaryFile(suffix=".mci", delete=True, mode="w+") as temp:
        temp.writelines(phantom_mci)
        temp.flush()  # Ensure the data is written to the file

        try:
            result = subprocess.run(["./mcml", temp.name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            output_lines = result.stdout.decode().split('\n')  # Split stdout into lines

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

    return Rcd, Tcd, Abs, Tc

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



def evaluate_task(task, toolbox, x1, x2, x3, x4):
    pop = task[0]  # Get the population from the task tuple

    for child1, child2 in zip(pop[::2], pop[1::2]):
        if np.random.rand() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in pop:
        if np.random.rand() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        fit_value, = fit  # Unpack the tuple
        ind.fitness.values = (fit_value,)

    return pop,



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
    #print("", lines[0][0], float(lines[0][0])) 

    g_values = float(lines[0][0])
    a_values =  float(lines[1][0])
    tau_values = float(lines[2][0])
    d = float(lines[3][0])  # Assuming d is a single value on its line
    Rcd, Tcd, Tc = lines[4]

    #print("",Rcd, Tcd, Tc)

    return g_values, a_values, tau_values, d, float(Rcd), float(Tcd), float(Tc)


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


def main():
    g_values, a_values, tau_values, d, Rcd_meas, Tcd_meas, Tc_meas = read_input_file("input_gene.dat")

    x1, x2, x3, x4 = Rcd_meas, Tcd_meas, 1 - (0.027+Rcd_meas+Tcd_meas+Tc_meas), Tc_meas
    mua_o, mus_o = tau_values*(1 - a_values)/d, a_values*tau_values/d
    g_o = g_values
    
    print("Initial values: mua = {:.4f}, mus = {:.4f}, g = {:.4f}".format(mua_o, mus_o, g_o))

    try:
        with open("phantom.mci", "r") as file:
            phantom_mci = file.readlines()
    except IOError:
        print("Error: unable to open phantom.mci for reading")
        return

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Establecer el rango alrededor de mua_o y mus_o para la generación inicial de genes
    range_mua = mua_o * 0.1  # 10% of mua_o
    range_mus = mus_o * 0.1  # 10% of mus_o

    #toolbox.register("attr_float", lambda: random.uniform(max(0, mua_o - range_mua), mua_o + range_mua))
    #toolbox.register("attr_float_g", lambda: random.uniform(max(0, mus_o - range_mus), mus_o + range_mus))
    toolbox.register("attr_float", lambda: random.uniform(0, 1.1*mua_o))
    toolbox.register("attr_float_g", lambda: random.uniform(0, 1.1*mus_o))
    toolbox.register("attr_float_gg", lambda: random.uniform(0, 1))  # For g
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_float, toolbox.attr_float_g, toolbox.attr_float_gg), n=1)

    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective_function, x1=x1, x2=x2, x3=x3, x4=x4, d=d, phantom_mci=phantom_mci)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    #toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selRoulette)  # Selección por ruleta
    #toolbox.register("select", tools.selStochasticUniversalSampling)  # Selección estocástica universal

    pop = toolbox.population(n=2)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()

    NGEN = 2

    tasks = [(toolbox.clone(pop),) for _ in range(NGEN)]

    results = map(lambda task: evaluate_task(task, toolbox, x1, x2, x3, x4), tasks)
    best_individuals = []  # List to store the best individuals from each generation

    for gen, result in enumerate(results):
        remaining_gens = NGEN - gen - 1
        print(f"Generation {gen} completed. Remaining generations: {remaining_gens}")

        pop[:] = result[0]  # Update the population with the results
        # Ensure all individuals are evaluated
        invalids = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitnesses):
            fit_value, = fit  # Unpack the tuple
            ind.fitness.values = (fit_value,)

        # Filter individuals with NaN or None fitness value before compiling stats and selecting the best one
        valid_individuals = [ind for ind in pop if ind.fitness.valid and np.isfinite(ind.fitness.values[0])]

        if not valid_individuals:
            print("No valid individuals in this generation.")
        else:
            # Compile and record stats
            record = stats.compile(valid_individuals)
            logbook.record(gen=gen, **record)
            print(logbook.stream)

            best_individual = tools.selBest(valid_individuals, k=1)[0]
            best_individuals.append(best_individual)  # Store the best individual from this generation

            relative_error = objective_function(best_individual, x1, x2, x3, x4, d, phantom_mci.copy())
            print("The optimized parameters are:", best_individual)
            print("The relative error is:", relative_error)
            print("______________________________#################")


    # Find the best individual across all generations
    best_estimate = min(best_individuals, key=lambda ind: ind.fitness.values[0])

    print("The overall best estimate is:", best_estimate)
    #mua_best, mus_best, g_best = best_estimate
    objective_function(best_estimate, x1, x2, x3, x4, d, phantom_mci)
    write_best_estimate(best_estimate, x1, x2, x3, x4, d, phantom_mci)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
