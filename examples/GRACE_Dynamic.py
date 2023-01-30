import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from tqdm import tqdm

import GCL.augmentors as A
import GCL.losses as L
from GCL.eval import get_split, LREvaluator


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        # 维度为32
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        # 增强的图1 和图2
        # X是一个tensor,其维度为2708*1433
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        # z是节点的向量表示，经过映射器，转化为节点数*维度的tensor
        # z是原始图的映射
        # z1是增强1的映射，z2是增强2的映射
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


# def weight_init(m):
#     if isinstance(m,torch.nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=1.0)

def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    # 数据增强
    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    # 卷积模块
    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    # 映射模块
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    # contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    # contrast_model = DualBranchContrast(loss=L.HardnessInfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    # contrast_model = DualBranchContrast(loss=L.HardnessInfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    # contrast_model = DualBranchContrast(loss=L.RingLoss(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    contrast_model = L.DynamicLoss(tau=0.2, g=data).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    max_mif1 = 0.0
    max_maf1 = 0.0
    with tqdm(total=300, desc='(T)') as pbar:
        for epoch in range(1, 301):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch % 10 == 0:
                test_result = test(encoder_model, data)
                # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
                max_mif1 = max(test_result["micro_f1"], max_mif1)
                max_maf1 = max(test_result["macro_f1"], max_maf1)
    print('max_mif1:')
    print(max_mif1)
    print('\n')
    print('max_maf1:')
    print(max_maf1)
    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
