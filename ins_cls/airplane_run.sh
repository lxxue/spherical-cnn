TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
				       python3 ../scripts/train.py \
                               @../params/model-64.txt \
                               @../params/airplane-64.txt \
                               @../params/training.txt \
                               --dset_dir ~/data/airplane \
                               --logdir ./airplane_log \
                               --run_id airplane
