for i in `seq 0 9`
do
    TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
                           python3 ../scripts/finetune.py \
                                   @../params/few_all.txt \
                                   @../params/few_retraining.txt \
                                   --dset_dir ~/data/s2cnn2/few_m10/$i/ \
                                   --logdir ./few_log/$i \
                                   --run_id few_$i \
                                   --ckpt "/home/lixin/Documents/spcnn_s2cnn/my_repo/spherical-cnn/baseline/30_log/best.ckpt"
done
