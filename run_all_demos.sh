#!/bin/bash 
for file in `ls -d mriData/*/`; do
   echo [$file --- Toeplitz];
   ./mriSolver -idir $file -cg_num 10 -toeplitzGridding -ntime_segs 8 -gpu_id 0 -gridOS_Q 1.250 -gridOS_FH 1.375;
done

