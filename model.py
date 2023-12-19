import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        for param in self.parameters():
            nn.init.normal_(param, std=0.01)

    def forward(self, user_id, item_id):
        """
        @param user_id: torch.LongTensor of shape (batch_size, 1)
        @param item_id: torch.LongTensor of shape (batch_size, 1)
        """
        P_u = self.user_embedding(user_id)
        Q_i = self.item_embedding(item_id)
        b_u = self.user_bias(user_id).flatten()
        b_i = self.item_bias(item_id).flatten()
        outputs = (P_u * Q_i).sum(axis=1) + b_u + b_i
        return outputs.flatten()
    

# refer https://discuss.pytorch.org/t/rmse-loss-function/16540/3
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class NeuMF(nn.Module):
    def __init__(self, embedding_dims, num_users, num_items, nums_hiddens, **kwargs):
        super().__init__()
        self.P = nn.Embedding(num_users, embedding_dims)
        self.Q = nn.Embedding(num_items, embedding_dims)
        self.U = nn.Embedding(num_users, embedding_dims)
        self.V = nn.Embedding(num_items, embedding_dims)
        self.mlp = nn.Sequential()
        self.mlp.add_module(
            'concat_layer', 
            nn.Linear(
                embedding_dims * 2, 
                nums_hiddens[0]
            )
        )
        self.mlp.add_module('concat_act', nn.ReLU())
        for i in range(len(nums_hiddens) - 1):
            self.mlp.add_module(
                f'linear{i}', 
                nn.Linear(
                    nums_hiddens[i], 
                    nums_hiddens[i+1]
                )
            )
            self.mlp.add_module(f'act{i}', nn.ReLU())
        self.prediction_layer = nn.Linear(
            nums_hiddens[-1] + embedding_dims, 1, bias=False
        )

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf

        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat([p_mlp, q_mlp], axis=-1))
        logit = self.prediction_layer(
            torch.cat([gmf, mlp], axis=-1)
        )
        return logit


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos, neg):
        loss = -F.logsigmoid(pos - neg)
        return loss.mean()
    

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pos, neg, margin=1.0):
        loss = F.relu(neg - pos + margin)
        return loss.mean()