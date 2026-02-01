#! /bin/bash



set -e



procedure_dir=$1



source ${procedure_dir}/scripts/CONFIG ${procedure_dir}



in_dir=${PROCEDURE_DIR}/results/1_input
out_dir=${PROCEDURE_DIR}/results/2_MD

${PROCEDURE_DIR}/scripts/run_GROMACS.sh ${in_dir}  \
                                        ${out_dir}
