#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./generate.sh [make|run]"
    exit 1
fi

# params and its operations
if [ $1 == "make" ]; then
    make
elif [ $1 == "run" ]; then
    # change the specs when needed, like mem and time
    srun -c 1 -t 10:00 -p gpu --mem=1G -o out.txt -e err.txt ./resultexe data_100k_arcmin.dat rand_100k_arcmin.dat omega.out
else
    echo "Unknown command: $1"
    echo "Usage: ./generate.sh [make|run]"
    exit 1
fi