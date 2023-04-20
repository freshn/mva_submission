#!/usr/bin/env bash
export TORCH_DISTRIBUTED_DEBUG=INFO 
export NCLL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
GPU_NUM=2

###############################
# Step 5: To generate the predictions for submission, the result will be saved in results.bbox.json.
###############################
echo "###############################"
echo "Step 5: To generate the predictions for submission, the result will be saved in results.bbox.json."
echo "###############################"
bash tools/dist_test.sh configs/mva2023_baseline_r101/centernet_r101_140e_coco_inference.py ckpt/baseline_r101.pth \
2 --format-only --eval-options jsonfile_prefix=results

_time=`date +%Y%m%d%H%M`
mkdir -p submit/${_time}
SUBMIT_FILE=`echo ./submit/${_time}/results.json`
SUBMIT_ZIP_FILE=`echo ${SUBMIT_FILE//results.json/submit.zip}`
mv ./results.bbox.json $SUBMIT_FILE
zip $SUBMIT_ZIP_FILE $SUBMIT_FILE
