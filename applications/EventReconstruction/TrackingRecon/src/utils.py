import numpy as np
import mindspore
import random
import os, psutil
import sys
from datetime import datetime
import argparse, json, time

# get memory usage for ppid and pid
def all_mem(ppid=1):
    if(ppid==1):
        parent = psutil.Process(os.getppid())
        mem = parent.memory_info().rss/1024./1024./1024
        for child in parent.children(recursive=True):
            mem += child.memory_info().rss/1024./1024./1024
        return mem
    else:
        mem = psutil.Process(os.getpid()).memory_info().rss/1024./1024./1024
        return mem

def process_bar_train(num, total, dt, loss, acc, Type=''):
    rate = float(num)/total
    ratenum = int(50*rate)
    estimate = dt/rate*(1-rate)
    mem = all_mem()
    ava = psutil.virtual_memory().available/1024/1024/1024
    r = '\r{} [{}{}]{}/{} - used {:.1f}s / left {:.1f}s / loss {:.10f} / acc {:.4f} / use {:.1f}G / ava {:.1f}G'.format(Type, '*'*ratenum,' '*(50-ratenum), num, total, dt, estimate, loss, acc, mem, ava)
    sys.stdout.write(r)
    sys.stdout.flush()

def process_bar(num, total, dt):
    import psutil
    rate = float(num)/total
    ratenum = int(50*rate)
    estimate = dt/rate*(1-rate)
    ava = psutil.virtual_memory().available/1024/1024/1024
    r = '\r[{}{}]{}/{} - used {:.1f}s / left {:.1f}s / total {:.1f}s / ava {:.1f}G'.format(f'*'*ratenum,' '*(50-ratenum), num, total, dt, estimate, dt/rate, ava)
    sys.stdout.write(r)
    sys.stdout.flush()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(1)
    print(f"[initial seed] >>> seed num set to {seed}, PID:{os.getpid()}")

def init_device(config):
    mindspore.set_context(mode=config['graph_mode'])
    mode = 'dynamic' if config['graph_mode']==1 else 'static'
    print(f'the compilation framework is set to {mode} graph mode')
    if(config['gpu']=='CPU'):
        mindspore.context.set_context(device_target="CPU")
    else:
        mindspore.context.set_context(device_target="GPU", device_id=config['gpu'])
    seed_everything(config['seed_num'])
    print(f'use device {mindspore.context.get_context("device_target")}')

def mkdir(config):
    new_path = './log/'+config['exp_name']
    folder = os.path.exists(new_path)
    if not folder:
        os.popen(f'mkdir -p {new_path}')
        os.popen(f'mkdir -p {new_path}/plot')
    return new_path

def initial(config):
    device = init_device(config)
    logdir = mkdir(config)
    return logdir


def build_dataset(src, dst, node_feat, edge_feat, edge_label):
    N = len(src)
    gap1 = int(N/7*5)
    gap2 = int(N/7*6)
    train_set = {
            'src'        : src[:gap1],
            'dst'        : dst[:gap1],
            'node_feat'  : node_feat[:gap1],
            'edge_feat'  : edge_feat[:gap1],
            'edge_label' : edge_label[:gap1],
            'len'        : len(src[:gap1]),
            }
    test_set = {
            'src'        : src[gap1:gap2],
            'dst'        : dst[gap1:gap2],
            'node_feat'  : node_feat[gap1:gap2],
            'edge_feat'  : edge_feat[gap1:gap2],
            'edge_label' : edge_label[gap1:gap2],
            'len'        : len(src[gap1:gap2])
            }
    apply_set = {
            'src'        : src[gap2:],
            'dst'        : dst[gap2:],
            'node_feat'  : node_feat[gap2:],
            'edge_feat'  : edge_feat[gap2:],
            'edge_label' : edge_label[gap2:],
            'len'        : len(src[gap2:])
            }
    return train_set, test_set, apply_set

def data_shuffle(dataset):
    shuffled_indices = np.random.permutation(dataset['len'])
    dataset['src'] = dataset['src'][shuffled_indices]
    dataset['dst'] = dataset['dst'][shuffled_indices]
    dataset['node_feat'] = dataset['node_feat'][shuffled_indices]
    dataset['edge_feat'] = dataset['edge_feat'][shuffled_indices]
    dataset['edge_label'] = dataset['edge_label'][shuffled_indices]
    dataset['shuffle'] = shuffled_indices

def batch(dataset, bs=64, shuffle=True, drop_last=False):
    if(shuffle):
        data_shuffle(dataset)
    else:
        dataset['shuffle'] = np.arange(dataset['len'])

    N_bs = int(dataset['len']/bs)+1
    if(drop_last):
        N_bs = N_bs-1

    mini_bs = [0]
    for i in range(N_bs):
        if(dataset['len']>bs*i+bs):
            mini_bs.append(bs*i+bs)
        else:
            mini_bs.append(dataset['len'])

    index = []
    bs_src = []
    bs_dst = []
    node_feat = []
    edge_feat = []
    edge_label = []
    st = time.time()
    for i in range(N_bs):
        sub_node_feat  = np.concatenate([dataset['node_feat'][i] for i in range(mini_bs[i], mini_bs[i+1])])
        node_feat.append(sub_node_feat)
        sub_edge_feat  = np.concatenate([dataset['edge_feat'][i] for i in range(mini_bs[i], mini_bs[i+1])])
        edge_feat.append(sub_edge_feat)
        sub_edge_label = np.concatenate([dataset['edge_label'][i] for i in range(mini_bs[i], mini_bs[i+1])])
        edge_label.append(sub_edge_label)
        index.append([dataset['shuffle'][i] for i in range(mini_bs[i], mini_bs[i+1])])        
        #if(i%100==0 or i==N_bs-1):
        #    process_bar(i+1, N_bs, time.time()-st)
    #print()
    shift = []
    shift_index = []
    st = time.time()
    for i in range(N_bs):
        sub_bs_src = []
        sub_bs_dst = []
        sub_shift = []
        sub_shift_index = []
        temp_node = 0
        temp_edge = 0
        sub_bs = mini_bs[i+1]-mini_bs[i]
        for k in range(sub_bs):
            if(k==0):
                sub_bs_src.append(dataset['src'][i*sub_bs+k])
                sub_bs_dst.append(dataset['dst'][i*sub_bs+k])
                sub_shift.append(0)
                sub_shift_index.append(0)
            else:                
                temp_node += len(dataset['node_feat'][i*sub_bs+k-1])
                sub_shift.append(temp_node)
                temp_edge += len(dataset['edge_feat'][i*sub_bs+k-1])
                sub_shift_index.append(temp_edge)
                sub_src = dataset['src'][i*sub_bs+k] + temp_node
                sub_bs_src.append(sub_src)
                sub_dst = dataset['dst'][i*sub_bs+k] + temp_node
                sub_bs_dst.append(sub_dst)
        bs_src.append(np.concatenate([sub_bs_src[j] for j in range(sub_bs)]))
        bs_dst.append(np.concatenate([sub_bs_dst[j] for j in range(sub_bs)]))
        shift.append(sub_shift)
        shift_index.append(sub_shift_index)
        #if(i%100==0 or i==N_bs-1):
        #    process_bar(i+1, N_bs, time.time()-st)
    #print()
    batched_dataset = {
        'bs_src'      : np.array(bs_src, dtype=object),
        'bs_dst'      : np.array(bs_dst, dtype=object),
        'node_feat'   : np.array(node_feat, dtype=object),
        'edge_feat'   : np.array(edge_feat, dtype=object),
        'edge_label'  : np.array(edge_label, dtype=object),
        'len'         : N_bs,
        'mini_bs'     : np.array(mini_bs),
        'shift'       : shift,
        'shift_index' : shift_index,
        'index'       : index,
        } 

    return dataset, batched_dataset


