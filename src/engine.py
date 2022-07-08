import torch
from torch.autograd.profiler import record_function
from torch.profiler import ExecutionGraphObserver
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer, _time
from metrics import MetronAtK
import tempfile


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

        # Timing
        self.event_start = torch.cuda.Event(enable_timing=True)
        self.event_end = torch.cuda.Event(enable_timing=True)
        self.time_fwd = 0
        self.time_bwd = 0
        self.global_batch_count = 0

        if self.config['collect_execution_graph']:
            fp = tempfile.NamedTemporaryFile('w+t', prefix='/tmp/pytorch_execution_graph_', suffix='.json', delete=False)
            fp.close()
            self.eg = ExecutionGraphObserver()
            self.eg.register_callback(fp.name)

    def warmup(self, user_indices, item_indices, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        if self.config['use_cuda']:
            B = user_indices.shape[0]
            offsets = torch.tensor(range(B+1), dtype=torch.int32).cuda()
            user_indices, items_indices, ratings = user_indices.int().view(-1).cuda(), item_indices.int().view(-1).cuda(), ratings.cuda()
        ratings_pred = self.model(user_indices, items_indices, offsets if self.config['use_cuda'] else None)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        loss = loss.item()
        return loss

    def train_single_batch(self, user_indices, item_indices, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        self.global_batch_count += 1
        with record_function("## Forward ##"):
            if self.config['use_cuda']:
                with record_function("module::forward_pass::distribute_emb_data"):
                    B = user_indices.shape[0]
                    offsets = torch.tensor(range(B+1), dtype=torch.int32).cuda()
                    user_indices, items_indices, ratings = user_indices.int().view(-1).cuda(), item_indices.int().view(-1).cuda(), ratings.cuda()
            t1 = _time(self.config['use_cuda'])
            if self.config['use_cuda']:
                self.event_start.record()
            ratings_pred = self.model(user_indices, items_indices, offsets if self.config['use_cuda'] else None)
            if self.config['use_cuda']:
                self.event_end.record()
            t2 = _time(self.config['use_cuda'])
        self.time_fwd += self.event_start.elapsed_time(self.event_end) * 1.e-3 if self.config['use_cuda'] else (t2 - t1)
        with record_function("## Backward ##"):
            t1 = _time(self.config['use_cuda'])
            if self.config['use_cuda']:
                self.event_start.record()
            loss = self.crit(ratings_pred.view(-1), ratings)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            if self.config['use_cuda']:
                self.event_end.record()
            t2 = _time(self.config['use_cuda'])
        self.time_bwd += self.event_start.elapsed_time(self.event_end) * 1.e-3 if self.config['use_cuda'] else (t2 - t1)

        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        self.model.train()
        total_loss = 0
        total_time = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.warmup(user, item, rating)
            if batch_id >= self.config['warmup_batches']:
                break

        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()

            if self.global_batch_count == 0 and self.config['collect_execution_graph']:
                self.eg.start()

            if batch_id < self.config['warmup_batches']:
                loss = self.warmup(user, item, rating)
            else:
                loss = self.train_single_batch(user, item, rating)
                if (batch_id + 1) % self.config['print_freq'] == 0:
                    print('[Training Epoch {}] Batch {}, Loss {}, Time {}'.format(epoch_id, batch_id + 1, loss, self.time_fwd + self.time_bwd - total_time))
                total_loss += loss
                total_time = self.time_fwd + self.time_bwd
                if self.global_batch_count >= self.config['num_batches']:
                    break

            if self.global_batch_count == 0 and self.config['collect_execution_graph']:
                self.eg.stop()
                self.eg.unregister_callback()
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config['use_cuda']:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()
            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)
            if self.config['use_cuda']:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
