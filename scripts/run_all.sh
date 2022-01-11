#!/bin/bash
#$ -wd /path/to/qsub-test # current directory for running
#$ -P alb.prjc
#$ -N alb
#$ -q short.qc
#$ -t 1-10 # number of subjects

# to submit: 
# https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/cluster-usage
# qsub run_all.sh

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

# https://www.medsci.ox.ac.uk/divisional-services/support-services-1/bmrc/python-on-the-bmrc-cluster

module load Python/3.7.2-GCCcore-8.2.0

source /path/to/projectA-${MODULE_CPU_TYPE}/bin/activate


### VARIABLES TO SET BEFORE RUNNING
# directory containing subdirectories named fter subject IDs that contain the timeseries and surface files
root_dir = 
# directory where all intermediate files and the final output will be saved
output_dir = 
# list of IDs of subjects to include in the analyses
# $SGE_TASK_ID
# For example: 
sub_list=./data2/manifest.txt
subj_id=$(sed -n "${SGE_TASK_ID}" $sub_list)

#-----------------------------------------------------------------------------------

python gradient_pctiles.py -s $subj_id -o $output_dir -r $root_dir

# python peak_dist.py -s $subj_id -o $output_dir -r $root_dir