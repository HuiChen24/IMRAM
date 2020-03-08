data_name=f30k_precomp
Batch_size=128
DATA_PATH=../../../data
MODEL_DIR=checkpoint/f30k_scan/

GPU_ID="1,2"
iteration_step=1
lambda_softmax=9
model_mode=image_IMRAM #full_IMRAM, text_IMRAM, image_IMRAM

echo "---------------training--------------"
logger_name=${MODEL_DIR}/gpus_${model_mode}_steps_${iteration_step}_softmax_${lambda_softmax}

model_name=${logger_name}

echo ${logger_name}

python train_gpus.py --gpuid ${GPU_ID} --batch_size ${Batch_size} --data_path ${DATA_PATH} --data_name ${data_name} --vocab_path vocab --logger_name ${logger_name} --model_name ${model_name} --max_violation --bi_gru --agg_func=Mean --lambda_softmax=${lambda_softmax} --model_mode ${model_mode} --iteration_step ${iteration_step} --no_IMRAM_norm

echo "---------------evaluation--------------"
MODEL_PATH=${model_name}/model_best.pth.tar
echo ${MODEL_PATH}

SPLIT=test

python test_gpus.py --gpuid ${GPU_ID} --model_path ${MODEL_PATH} --data_path ${DATA_PATH} --split ${SPLIT}
