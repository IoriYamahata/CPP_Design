#! /bin/bash



set -e



procedure_dir=$1




source ${procedure_dir}/scripts/CONFIG ${procedure_dir}



in_dir=${PROCEDURE_DIR}/data/design
out_dir=${PROCEDURE_DIR}/results/1_input

${PROCEDURE_DIR}/scripts/make_GROMACS_input.sh ${in_dir}  \
                                               ${out_dir}
