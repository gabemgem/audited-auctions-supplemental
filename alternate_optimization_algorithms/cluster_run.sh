#!/bin/bash -l

# example usage:
# qsub cluster_run.sh data/a_single_modal_normal_distribution.csv 1

# job submission directives
#$ -P attestia
#$ -N auction-optimization
#$ -M gmaayan@bu.edu
#$ -m ea
#$ -o output/logs/output.log
#$ -e output/logs/output.err

# Submit an array job with 10 tasks
#$ -t 1-7

# multiple slots for shared memory applications
#$ -pe omp 1

filename=$(basename "$1")

datatag="${filename:0:1}"

runtag="${datatag}_${2}_${SGE_TASK_ID}"

num_generations=600

if [ "$2" -gt 1 ]; then
    num_generations=4000
fi

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "Run Tag : $runtag"
echo "=========================================================="

# specify python modules
module load python3/3.10.5

pip install pygad

python Collateralized_Auction_genetic_script.py --scc-it $SGE_TASK_ID --data $1 --polynomial-degree $2 --seed 1234 --id $runtag --num-generations $num_generations