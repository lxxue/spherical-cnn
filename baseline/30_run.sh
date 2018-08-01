TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=3 \
				       python3 ../scripts/train.py \
                               @../params/model-64.txt \
                               @../params/m40-64.txt \
                               @../params/training.txt \
                               --dset_dir ~/data/m30 \
                               --logdir ./m30_log \
                               --run_id m30
