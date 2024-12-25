import os
import time
import random
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pynvml

parser = argparse.ArgumentParser(description='')
parser.add_argument('--run', action='store_true', help="")
parser.add_argument('--double_check', action='store_true', help="double check before creating tensors")
parser.add_argument('--waittime', type=float, default=30, help="wait time in seconds")
parser.add_argument('--mem_ratio', type=float, default=0.85, help="memory ratio")
parser.add_argument('--sleep_ratio', type=float, default=0.02, help="sleep ratio")
parser.add_argument('--fix', action='store_true', help="fix the memory size")
parser.add_argument('--ports', type=int, default=29291, help="sleep ratio")
args = parser.parse_args()

def get_avaliable_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)        # 0表示第一块显卡
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    ava_mem=round(meminfo.free/1024**2)
    print('current available video memory is' +' : '+ str(round(meminfo.free/1024**2)) +' MIB')
    return ava_mem

def check_mem():
    mems = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).readlines()
    gpu_info = []
    for item in mems:
        info = item.split(',')
        gpu_info.append({"total": int(info[0]), "used": int(info[1])})
    gpu_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if gpu_device is not None:
        gpu_device = gpu_device.split(',')
        avail_gpu_info = [gpu_info[int(gpu)] for gpu in gpu_device]
    else:
        avail_gpu_info = gpu_info
    return avail_gpu_info

import threading, multiprocessing

def loop():
    x = 0
    while True:
        x = x ^ 1