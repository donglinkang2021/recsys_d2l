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