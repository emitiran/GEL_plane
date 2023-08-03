import numpy as np
import subprocess
from deap import base, creator, tools, algorithms
import multiprocessing
import random

import tempfile
# Define la función black_box
def black_box(inputs):
    """
    Función que realiza los cálculos y genera los valores de salida (Rcd, Tcd, Abs, Tc) a partir de los parámetros de entrada.

    Args:
        mua (float): Coeficiente de absorción.
        mus (float): Coeficiente de scattering reducido.
        g (float): Parámetro de anisotropía.
        a (float): Parámetro a.
        tau (float): Parámetro tau.

    Returns:
        tuple: Tupla con los valores de salida (Rcd, Tcd, Abs, Tc).
    """
    mua, mus, g, d = inputs
    try:
        with open("phantom.mci", "r") as file:
            lines = file.readlines()
    except IOError:
        print("Error: unable to open phantom.mci for reading")
        return float('inf'), float('inf'), float('inf'), float('inf')

    # Read the values from line 19
    t1, _, _, _, t5, *_ = lines[19].split()
    t5 = d
    # Write back the values to line 19, replacing t2, t3, and t4
    lines[19] = "{}\t{}\t{}\t{}\t{}\n".format(t1, mua, mus, g, t5)

    # Use a temporary file instead of "phantom2.mci"
    with tempfile.NamedTemporaryFile(suffix=".mci", delete=True, mode="w+") as temp:
        temp.writelines(lines)
        temp.flush()  # Ensure the data is written to the file

        try:
            result = subprocess.run(["./mcml", temp.name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            output_lines = result.stdout.decode().split('\n')  # Split stdout into lines
            #print(output_lines)
            Rcd_line = output_lines[33]
            Abs_line = output_lines[34]
            Tcd_line = output_lines[35]
            Tc_line = output_lines[36]

            Rcd = float(Rcd_line.split()[0])
            Tcd = float(Tcd_line.split()[0])
            Abs = float(Abs_line.split()[0])
            Tc = float(Tc_line.split()[0])
        except subprocess.CalledProcessError:
            return float('inf'), float('inf'), float('inf'), float('inf')

    return Rcd, Tcd, Abs, Tc



def objective_function(individual, x1, x2, x3, x4, d):
    #x1: Rcd
    #x2: Tad
    #x3: Abs
    #x4: Tc
    mua, mus, g = individual  # added "g"
    Rcd, Tcd, Abs, Tc = black_box((mua, mus, g, d))  # included "g" in the function call
  
    print("Rcd_o: {:.4f}, Tcd_o: {:.4f}, Tc_o: {:.4f}, A_o: {:.4f}".format(x1, x2, x4, x3))
    print("Rcd  : {:.4f}, Tcd  : {:.4f}, Tc  : {:.4f}, A  : {:.4f}".format(Rcd, Tcd, Tc, Abs))
    print("mua: {:.4f}, mus: {:.4f}, g: {:.4f}".format(mua, mus, g))
    
    #return (x1 - Rcd) ** 2 + (x2 - Tcd) ** 2 + (x3 - Abs) ** 2 + (x4 - Tc) ** 2
    
    out_val =  np.abs(x1 - Rcd)/(Rcd+1e-9)
    out_val = out_val + np.abs(x2 - Tcd)/(Tcd + 1e-9)
    #out_val = out_val + np.abs(x3 - Abs)/(Abs + 1e-9)
    out_val = out_val + np.abs(x4 - Tc)/(Tc + 1e-9)
    print("Error relativo: {:.4f}".format(out_val))
    return out_val

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
        ind.fitness.values = (fit,)  # Assign fitness as a tuple

    return pop,

def main():
    #x1: Rcd
    #x2: Tad
    #x3: Abs
    #x4: Tc

    x1, x2, x3, x4 = 0.40, 0.35, 0.21, 0.0009
    mua_o, mus_o = 0.06, 20.11
    g_o = 0.65  # added "g_o" as the initial value for "g"
    d =.34

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", lambda: random.uniform(mua_o, mus_o))
    toolbox.register("attr_float_g", lambda: random.uniform(0, 1))  # added range for "g"
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float_g), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective_function, x1=x1, x2=x2, x3=x3, x4=x4, d=d)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=100)


    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()

    NGEN = 100

    tasks = [(toolbox.clone(pop),) for _ in range(NGEN)]  # Create a tuple with the population for each task

    results = map(lambda task: evaluate_task(task, toolbox, x1, x2, x3, x4), tasks)

    for gen, result in enumerate(results):
        remaining_gens = NGEN - gen - 1  # Calcular la cantidad de generaciones restantes
        print(f"Generación {gen} completada. Generaciones restantes: {remaining_gens}")

        pop[:] = result[0]  # Update the population with the results

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        best_individual = tools.selBest(pop, k=1)[0]
        print("The optimized parameters are:", best_individual)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
