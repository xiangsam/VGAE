import torch
import torch.nn as nn
from torchsummary import summary
from einops import rearrange
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, drop_ratio=0.):
        super().__init__()
        self.dropout = nn.Dropout(drop_ratio)
        self.conv = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self, x, A):
        # x [N D]
        # A [N N]
        h = A @ self.conv(x)
        h = self.dropout(h)
        
        return h

class VGAE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, z_dim, drop_ratio=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.gcn1 = GCN(embed_dim, hidden_dim)
        self.gcn21 = GCN(hidden_dim, z_dim) # for mu
        self.gcn22 = GCN(hidden_dim, z_dim) # for log_var
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_ratio)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            # std = torch.exp(logvar)
            eps = torch.normal(mean=0, std=1,
                            size=mu.shape).to(mu.device)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, x, A):
        h = self.act1(self.gcn1(x, A))
        mu = self.gcn21(h, A)
        logvar = self.gcn22(h, A)
        return mu, logvar
    
    def decode(self, z):
        z = self.dropout(z)
        # gene_A = self.act2(z @ z.T)
        gene_A = z @ z.T # important to performance
    
        return gene_A
    
    def loss(self, A, gene_A, mu, logvar):
        # BCE_loss = torch.sum(nn.NLLLoss(reduction='none')(gene_A.flatten(), A.flatten().long()), dim=-1)
        # BCE_loss = norm * F.binary_cross_entropy_with_logits(gene_A, A, pos_weight=pos_weight)
        # BCE_loss = F.binary_cross_entropy_with_logits(gene_A, A, pos_weight=pos_weight)
        BCE_loss = F.cross_entropy(gene_A, A)
        KL_loss = -0.5/ A.shape[0] * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))
        # KL_loss = -0.5 / A.shape[0] * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))
        # print(BCE_loss, KL_loss)
        return BCE_loss + KL_loss
    
    def forward(self, x, A_norm, A_label):
        mu, logvar = self.encode(x, A_norm)
        z = self.reparameterize(mu, logvar)
        gene_A = self.decode(z)
        loss = self.loss(A_label, gene_A, mu, logvar)
        return loss, gene_A

if __name__ == '__main__':
    model = VGAE(90, 32, 16).to('cuda:0')
    summary(model, [(90,), (2,)],device='cuda')

        