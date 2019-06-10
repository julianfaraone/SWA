python3 train.py --dir=train_log \
                 --dataset=imagenet \
                 --data_path=../../QPyTorch/examples/IBM8/data/imagenet12/ \
                 --model=resnet \
                 --epochs=10 \
                 --lr_init=0.001 \
                 --wd=0.0001 \
                 --swa \
                 --swa_start=2 \
                 --pretrained \
                 --swa_lr=0.00001
#                 --swa_lr=0.05


#                --lr_init=0.0005 \
