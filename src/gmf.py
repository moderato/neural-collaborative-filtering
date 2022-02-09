import torch
from engine import Engine
from utils import use_cuda
import table_batched_embeddings_ops

class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.use_cuda = config['use_cuda']

        if self.use_cuda:
            self.embedding_user = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
                1,
                self.num_users,
                self.latent_dim,
                optimizer=table_batched_embeddings_ops.Optimizer.SGD,
                learning_rate=1e-9,
                eps=None,
                stochastic_rounding=False,
            )
            self.embedding_item = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
                1,
                self.num_items,
                self.latent_dim,
                optimizer=table_batched_embeddings_ops.Optimizer.SGD,
                learning_rate=1e-9,
                eps=None,
                stochastic_rounding=False,
            )
        else:
            self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, offsets=None):
        if self.use_cuda:
            user_embedding = self.embedding_user(user_indices, offsets)
            item_embedding = self.embedding_item(item_indices, offsets)
            element_product = torch.mul(user_embedding.squeeze(1), item_embedding.squeeze(1))
        else:
            user_embedding = self.embedding_user(user_indices)
            item_embedding = self.embedding_item(item_indices)
            element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)