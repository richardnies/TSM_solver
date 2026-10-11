#!/bin/bash
# run the heat-flux check decks (a few seconds each on 4 MPI ranks)
cd "$(dirname "$0")/runs"
for d in */; do (cd $d && mpirun -n 4 /home/rnies/stella_COPY/stella CBC.in > stella.log 2>&1; echo "$d $(tail -n 1 CBC.fluxes)"); done
