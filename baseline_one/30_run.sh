TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
				       python3 ../scripts/train.py \
                               @../params_one/model-64.txt \
                               @../params_one/m30-64.txt \
                               @../params_one/training.txt \
                               --dset_dir ~/data/s2cnn/m30 \
                               --logdir ./m30_log \
                               --run_id m30
