#!/bin/sh

#PBS -q teslaq
#PBS -l walltime=0:0:10
#PBS -l select=1:ngpus=3,place=pack:excl

#qsub -q teslaq -l select=1:ngpus=3,place=pack:excl,walltime=0:0:10


cd $PBS_O_WORKDIR

qstat -f $PBS_JOBID >log

vnodes=$(qstat -f $PBS_JOBID|grep exec_vnode|sed -e 's/ *//')

gpu_nbr=$(echo $vnodes|sed 's/.*\[//'|sed 's/\].*//')
echo "GPU number from vnodes = $gpu_nbr"
echo

export PATH=$PATH:/opt/shared/nvidia/cuda/bin
export LD_LIBRARY_PATH=/opt/shared/nvidia/cuda/lib64:/opt/shared/nvidia/cuda/lib:$LD_LIBRARY_PATH

#touch ./123

./lapl3d
