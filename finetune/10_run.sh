TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=1 \
				       python3 ../scripts/finetune.py \
                               @../params/10_all.txt \
                               @../params/retraining.txt \
                               --dset_dir ~/data/s2cnn2/m10 \
                               --logdir ./10_log \
                               --run_id 10 \
                               --ckpt "/home/lixin/Documents/spcnn_s2cnn/my_repo/spherical-cnn/baseline/30_log/best.ckpt"
