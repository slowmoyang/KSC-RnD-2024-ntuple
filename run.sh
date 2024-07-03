#!/usr/bin/env bash
#SBATCH --job-name=KSC-RnD-2024.ntuple
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/run_%j.out
#SBATCH --error=./logs/run_%j.err

echo "START: $(date)"
echo "USER: $(whoami)"
echo "HOST: $(hostname)"
echo "CWD: $(pwd)"

source ./utils.sh

###############################################################################
#
###############################################################################
assert_var_defined SLURM_ARRAY_TASK_ID
assert_var_defined OFFSET
assert_var_defined DATASET



###############################################################################
#
###############################################################################
INDEX=$((${OFFSET} + ${SLURM_ARRAY_TASK_ID}))
assert_var_defined INDEX

#------------------------------------------------------------------------------
COUNTER=$(get_counter ${INDEX})
assert_var_defined COUNTER

STORE=/store/hep/users/slowmoyang/
assert_dir_exists STORE

INPUT_FILE=${STORE}/KSC-RnD-2024/dataset/${DATASET}/05-delphes/${COUNTER}/output_${INDEX}.root
assert_file_exists INPUT_FILE

OUTPUT_DIR=${STORE}/KSC-RnD-2024/ntuple/${DATASET}/${COUNTER}
mkdir -vp ${OUTPUT_DIR}
assert_dir_exists OUTPUT_DIR

OUTPUT_FILE=${OUTPUT_DIR}/output_${INDEX}.root
assert_file_not_exists OUTPUT_FILE

eval "$(micromamba shell hook --shell bash)"
micromamba activate KSC-RnD-2024-ntuple-py311

python ./ntuplise.py -i ${INPUT_FILE} -o ${OUTPUT_FILE}

###############################################################################
#
###############################################################################
echo "END: $(date)"
