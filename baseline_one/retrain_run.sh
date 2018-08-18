TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
				       python3 ../scripts/finetune.py \
                               @../params/model-64.txt \
                               @../params/m30-64.txt \
                               @../params/retraining.txt \
                               --dset_dir ~/data/m30 \
                               --logdir ./retrain_log \
                               --run_id retrain \
                               --ckpt "/home/lixin/Documents/mygithub/new_data2/spherical-cnn/baseline/retrain_log/best.ckpt"
