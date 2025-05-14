
conda activate cellpose_env

export LD_LIBRARY_PATH=/path/to/torch/library/:$LD_LIBRARY_PATH

python -m cellpose --train --use_gpu --dir /path/to/images/ --pretrained_model /path/to/pre_trained/model/ --mask_filter _seg.npy  --chan 1 --chan2 2 --learning_rate 0.1 --weight_decay 0.0001 --n_epochs 300 --verbose