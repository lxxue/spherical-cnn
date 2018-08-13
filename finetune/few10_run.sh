for i in `seq 0 9`
do
    TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
                   python3 ../scripts/finetune.py \
                           @../params/model-64.txt \
                           @../params/m10-64.txt \
                           @../params/retraining.txt \
                           --dset_dir ~/data/few_m10/few_m10_$i \
                           --logdir ./few_m10_log/$i \
                           --run_id few_m10_$i \
                           --ckpt "/home/lixin/Documents/mygithub/new_data2/spherical-cnn/baseline/m30_log2/best.ckpt"
done
