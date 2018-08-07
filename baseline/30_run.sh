TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
				       python3 ../scripts/train.py \
                               @../params/model-64.txt \
                               @../params/m30-64.txt \
                               @../params/training.txt \
                               --dset_dir ~/data/m30 \
                               --logdir ./m30_log2 \
                               --run_id m302
