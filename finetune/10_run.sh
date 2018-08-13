TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=1 \
				       python3 ../scripts/finetune.py \
                               @../params/model-64.txt \
                               @../params/m10-64.txt \
                               @../params/retraining.txt \
                               --dset_dir ~/data/m10 \
                               --logdir ./m10_log \
                               --run_id m10 \
                               --ckpt "/home/lixin/Documents/mygithub/new_data2/spherical-cnn/baseline/m30_log2/best.ckpt"
