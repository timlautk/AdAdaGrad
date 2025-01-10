# Logistic regression on MNIST
## norm test
fabric run main.py --devices=1 --num_workers=0 --optimizer=adagrad \
--batch_size=2 --max_test_micro_batch_size=60000 --max_batch_size=60000 \
--learning_rate=8e-3 --max_lr=8e-3 --decay_lr=False --max_samples=6000000 \
--model=logistic --dataset=mnist --test_type=norm --eta=1e-1

# CNN on MNIST
## norm test
fabric run main.py --devices=4 --num_workers=0 --optimizer=adagrad \
--batch_size=8 --max_test_micro_batch_size=15000 --max_batch_size=60000 \
--learning_rate=8e-3 --max_lr=8e-3 --decay_lr=False --max_samples=6000000 \
--model=cnn --dataset=mnist --test_type=norm --eta=1e-1


# ResNet-18 on CIFAR-10
## norm test
fabric run main.py --devices=4 --num_workers=0 --optimizer=adagrad \
--batch_size=8 --max_test_micro_batch_size=100 --max_micro_batch_size=128 --max_batch_size=50000 \
--max_samples=10000000 --warmup_samples=1000000 --lr_decay_samples=9000000 \
--learning_rate=5e-2 --min_lr=5e-3 --model=resnet18 --dataset=cifar10 --test_type=norm --eta=5e-1


# ResNet-50 on CIFAR-10
## norm test
fabric run main.py --devices=4 --num_workers=0 --optimizer=sgdm \
--batch_size=256 --max_test_micro_batch_size=64 --max_micro_batch_size=100 --max_batch_size=50000 \
--max_samples=10000000 --warmup_samples=1000000 --lr_decay_samples=9000000 \
--learning_rate=5e-1 --min_lr=5e-2 --model=resnet50 --dataset=cifar10 --test_type=norm --eta=2.5e-1


# ResNet-50 on CIFAR-100
## norm test
fabric run main.py --devices=4 --num_workers=0 --optimizer=sgdm \
--batch_size=256 --max_test_micro_batch_size=64 --max_micro_batch_size=100 --max_batch_size=50000 \
--max_samples=10000000 --warmup_samples=1000000 --lr_decay_samples=9000000 \
--learning_rate=5e-1 --min_lr=5e-2 --model=resnet50 --dataset=cifar100 --test_type=norm --eta=2.5e-1


# ResNet-101 on ImageNet
## norm test
fabric run main.py --devices=4 --num_workers=12 --precision=bf16-mixed --optimizer=sgdm \
--data_dir=/net/projects/timlautk/data/imagenet-gen/ \
--batch_size=256 --max_test_micro_batch_size=64 --max_micro_batch_size=512 --max_batch_size=111360 \
--max_samples=256233400 --warmup_samples=6405835 --lr_decay_samples=249827565 \
--learning_rate=2.5e0 --min_lr=2.5e-1 --weight_decay=1e-4 --gradient_accumulation_steps=1 \
--model=resnet101 --dataset=imagenet --test_type=norm --eta=1e-1



