import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

import torch
import torch.autograd.profiler as profiler
from torch.autograd.profiler import record_function
import argparse, graph_observer
from caffe2.python import core
core.GlobalInit(
    [
        "python",
        "--pytorch_enable_execution_graph_observer=true",
        "--pytorch_execution_graph_observer_iter_label=## BENCHMARK ##",
    ]
)

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'device_id': 7,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'device_id': 7,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disable CUDA')
    parser.add_argument('--profile', action='store_true', default=False,
                       help='enable autograd profiler')
    parser.add_argument('--collect-execution-graph', action='store_true', default=False,
                       help='collect execution graph')
    parser.add_argument("--batch-size", type=int, default=1024,
                       help='batch size')
    parser.add_argument("--num-epoch", type=int, default=20,
                       help='nb of epochs in loop to average perf')
    parser.add_argument("--num-batches", type=int, default=1e9,
                       help='nb of batches in loop to average perf')
    parser.add_argument("--print-freq", type=int, default=5,
                       help='print frequency')
    parser.add_argument("--engine-type", type=str, default='gmf',
                       help='nb of batches in loop to average perf')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='evaluate after training')
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # Load Data
    ml1m_dir = 'data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ml1m_rating)
    evaluate_data = sample_generator.evaluate_data

    # Specify the exact model
    if args.engine_type == 'gmf':
        config = gmf_config
        Engine = GMFEngine
    elif args.engine_type == 'mlp':
        config = mlp_config
        Engine = MLPEngine
    elif args.engine_type == 'neumf':
        config = neumf_config
        Engine = NeuMFEngine
    else:
        print("Unrecognized engine type! Quit...")
        exit()
    
    config['use_cuda'] = args.cuda
    config['num_epoch'] = args.num_epoch
    config['batch_size'] = args.batch_size
    config['num_batches'] = args.num_batches
    config['profile'] = args.profile
    config['collect_execution_graph'] = args.collect_execution_graph
    config['print_freq'] = args.print_freq
    engine = Engine(config)

    # Train and evaluate
    with profiler.profile(args.profile, use_cuda=args.cuda, use_kineto=True) as prof:
        class dummy_record_function():
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc_value, traceback):
                return False
        for epoch in range(config['num_epoch']):
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])

            with record_function("## BENCHMARK ##") if args.collect_execution_graph else dummy_record_function():
                engine.train_an_epoch(train_loader, epoch_id=epoch)

                if args.evaluate:
                    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
                    engine.save(config['alias'], epoch, hit_ratio, ndcg)

        time_fwd_avg = engine.time_fwd / engine.batch_count * 1000
        time_bwd_avg = engine.time_bwd / engine.batch_count * 1000
        time_total = time_fwd_avg + time_bwd_avg

        print("Overall per-batch training time: {:.2f} ms".format(time_total))

    if args.profile:
        with open("ncf_benchmark.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("ncf_benchmark.json")