TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
				       python3 ../scripts/train.py \
                               @../params/model-64.txt \
                               @../params/m40-64.txt \
                               @../params/training.txt \
                               --dset_dir ~/data/m10 \
                               --logdir ./m10_log \
                               --run_id m10
