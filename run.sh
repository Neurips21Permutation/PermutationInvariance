#!/bin/bash


### current
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
#caliban run --experiment_config config.json train.py
#caliban cloud --xgroup train --experiment_config config.json --gpu_spec 1xV100 train.py
#caliban cloud --xgroup train --experiment_config config.json --machine_type n1-standard-16 --gpu_spec 8xV100 train.py

#caliban run --experiment_config config_barrier.json barrier.py
#caliban cloud --xgroup barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier.py


#caliban run --experiment_config config_barrier.json barrier_resnet_sanity.py

#### V2
caliban run --experiment_config config_barrier.json barrier_instance_v2.py
#caliban cloud --xgroup barrier --experiment_config config_barrier.json --gpu_spec 1xP100 barrier_instance_v2.py
#




#caliban run --experiment_config config_barrier.json barrier_instance_v1.py
#caliban cloud --xgroup barrier --experiment_config config_barrier.json --gpu_spec 1xV100 barrier_instance_v1.py

