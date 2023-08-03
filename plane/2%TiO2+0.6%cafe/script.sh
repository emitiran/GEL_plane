#!/bin/bash

# Numero de veces que quieres correr el script
runs=16

# Directorio donde se guarda el archivo de salida
output_dir="."

for ((i=1; i<=runs; i++))
do
   echo "Run $i"
   caffeinate -i python3 plane_mcml.py
   mv "$output_dir/best_estimate.txt" "$output_dir/best_estimate_run_$i.txt"
done
