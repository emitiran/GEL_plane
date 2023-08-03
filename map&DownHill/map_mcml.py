import numpy as np
from scipy.optimize import curve_fit
import os
import subprocess
import concurrent.futures
import multiprocessing
import time
import matplotlib.pyplot as plt

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

    g_values = [float(value) for value in lines[0]]
    a_values = [float(value) for value in lines[1]]
    tau_values = [float(value) for value in lines[2]]
    d = float(lines[3][0])  # Assuming d is a single value on its line
    Rcd, Tcd, Tc = lines[4]

    #print("",Rcd, Tcd, Tc)

    return g_values, a_values, tau_values, d, float(Rcd), float(Tcd), float(Tc)


import tempfile

# Define la función black_box
def black_box(inputs):
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

    return mua, mus, g, Rcd, Tcd, Abs, Tc






import matplotlib.pyplot as plt
import numpy as np

def main():
    g_values, a_values, tau_values, d, Rcd_meas, Tcd_meas, Tc_meas = read_input_file("input_map.dat")
    inputs = [(tau*(1 - a)/d, a*tau/d, g, d) for a in a_values for tau in tau_values for g in g_values]
    tot_val = []
    start_time = time.time()

    for i, inp in enumerate(inputs, 1):
        elapsed_time = time.time() - start_time
        average_time = elapsed_time / i
        estimated_total_time = average_time * len(inputs)
        estimated_time_remaining = estimated_total_time - elapsed_time

        print(f"Completed {i}/{len(inputs)} ({100.0*i/len(inputs):.2f}%) in {elapsed_time:.2f}s, average: {average_time:.2f}s, estimated total time: {estimated_total_time:.2f}s, estimated time remaining: {estimated_time_remaining // 60:.0f}m {estimated_time_remaining % 60:.0f}s")

        try:
            mua, mus, g, Rcd, Tcd, Abs, Tc = black_box(inp)
            a = mus/(mus+mua)
            tau = d*(mus+mua)

            dif = (Rcd-Rcd_meas)/Rcd_meas + (Tcd-Tcd_meas)/Tcd_meas + (Tc-Tc_meas)/Tc_meas
            dif = abs(dif)

            tot_val.append((a, tau, mua, mus, g, d, Rcd, Tcd, Abs, Tc, dif))
        except Exception as exc:
            print(f'An exception occurred: {exc}')

    tot_val_sorted = sorted(tot_val, key=lambda x: x[0])

    with open('results.csv', 'w') as file:
        file.write('a,tau,mua,mus,g,d,Rcd,Tcd,Abs,Tc,dif\n')
        for val in tot_val_sorted:
            file.write(','.join(map(str, val)) + '\n')


if __name__ == '__main__':
    main()
