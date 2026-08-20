# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GetGraph(nn.Module):
    def forward(self, point_cloud):
        coords = point_cloud[..., :3]
        # Нормализация: центрируем и масштабируем
        coords = coords - coords.mean(dim=1, keepdim=True)
        max_dist = coords.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0] + 1e-8
        coords = coords / max_dist
        # Далее как было
        trans = coords.permute(0, 2, 1)
        inner = -2 * torch.matmul(coords, trans)
        sq = torch.sum(coords ** 2, dim=2, keepdim=True)
        sq_t = sq.permute(0, 2, 1)
        adj = sq + inner + sq_t
        return torch.exp(-adj)

    
#class GetGraph(nn.Module):
#    """Вычисление матрицы сходства по евклидовым расстояниям."""
#    def forward(self, point_cloud):
#        coords = point_cloud[..., :3]
#        trans = coords.permute(0, 2, 1)
#        inner = -2 * torch.matmul(coords, trans)
#        sq = torch.sum(coords ** 2, dim=2, keepdim=True)
#        sq_t = sq.permute(0, 2, 1)
#        adj = sq + inner + sq_t
#        return torch.exp(-adj)

class GetLaplacian(nn.Module):
    """Вычисление нормализованной матрицы Лапласа по формуле L = I - D^{-1/2} A D^{-1/2}."""
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, adj):
        if self.normalize:
            D = torch.sum(adj, dim=2)
            D_inv_sqrt = 1.0 / torch.sqrt(D + 1e-8)
            D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
            L = torch.eye(adj.size(1), device=adj.device).unsqueeze(0) - D_inv_sqrt @ adj @ D_inv_sqrt
        else:
            D = torch.sum(adj, dim=2)
            D = torch.diag_embed(D)
            L = D - adj
        return L

class GetFilter(nn.Module):
    """
    Свёртка на графе с использованием полиномов Чебышёва.
    """
    def __init__(self, Fin, K, Fout):
        super().__init__()
        self.Fin = Fin
        self.K = K
        self.Fout = Fout
        self.W = nn.Parameter(torch.Tensor(Fin * K, Fout))
        self.b = nn.Parameter(torch.Tensor(Fout))
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.b)
        self.relu = nn.ReLU()

    def forward(self, x, L):
        B, M, Fin = x.shape
        K = self.K
        if K == 1:
            polys = x.unsqueeze(0)
        else:
            x0 = x
            x1 = torch.matmul(L, x0)
            polys = torch.stack([x0, x1], dim=0)
            for _ in range(2, K):
                x2 = 2 * torch.matmul(L, x1) - x0
                polys = torch.cat([polys, x2.unsqueeze(0)], dim=0)
                x0, x1 = x1, x2
        polys = polys.permute(1, 2, 3, 0).reshape(B * M, Fin * K)
        out = torch.matmul(polys, self.W) + self.b
        out = self.relu(out)
        return out.reshape(B, M, self.Fout)

class RGCNN_Seg(nn.Module):
    """
    Модель RGCNN для сегментации.
    Параметры:
        vertice: число точек (M)
        F: список выходных фильтров для каждого свёрточного слоя
        K: список порядков полиномов
        num_classes: число классов сегментации (последний F должен быть равен num_classes)
        num_categories: число категорий объектов (для one-hot)
    """
    def __init__(self, vertice, F, K, num_classes, num_categories=1, regularization=0.0):
        super().__init__()
        assert len(F) == len(K)
        self.vertice = vertice
        self.F = F
        self.K = K
        self.num_classes = num_classes
        self.num_categories = num_categories
        self.regularization = regularization

        self.get_graph = GetGraph()
        self.get_laplacian = GetLaplacian(normalize=True)

        Fin_first = 3 + num_categories

        self.conv_layers = nn.ModuleList()
        for i, (f, k) in enumerate(zip(F, K)):
            Fin = Fin_first if i == 0 else F[i-1]
            self.conv_layers.append(GetFilter(Fin, k, f))

        if F[-1] != num_classes:
            self.fc_out = nn.Linear(F[-1], num_classes)
        else:
            self.fc_out = nn.Identity()

        self.regularizer_loss = nn.MSELoss()

    def forward(self, x, cat):
        coords = x[..., :3]
        adj = self.get_graph(coords)
        L = self.get_laplacian(adj)

        B, M, _ = x.shape
        cat_onehot = F.one_hot(cat, num_classes=self.num_categories).float()
        cat_onehot = cat_onehot.unsqueeze(1).expand(-1, M, -1)
        x = torch.cat([coords, cat_onehot], dim=2)

        for conv in self.conv_layers:
            x = conv(x, L)

        logits = self.fc_out(x)
        return logits