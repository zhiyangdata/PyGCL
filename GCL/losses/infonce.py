import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils

from .losses import Loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
	h1 = F.normalize(h1)
	h2 = F.normalize(h2)
	# print(h1)
	# print(h2)
	return h1 @ h2.t()


class InfoNCESP(Loss):
	"""
    InfoNCE loss for single positive.
    """

	def __init__(self, tau):
		super(InfoNCESP, self).__init__()
		self.tau = tau

	def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
		f = lambda x: torch.exp(x / self.tau)
		sim = f(_similarity(anchor, sample))  # anchor x sample
		assert sim.size() == pos_mask.size()  # sanity check

		neg_mask = 1 - pos_mask
		pos = (sim * pos_mask).sum(dim=1)
		neg = (sim * neg_mask).sum(dim=1)

		loss = pos / (pos + neg)
		loss = -torch.log(loss)

		return loss.mean()


class InfoNCE(Loss):
	def __init__(self, tau):
		super(InfoNCE, self).__init__()
		self.tau = tau

	def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
		# 误差计算
		# tau = 0.2
		# cos相似度
		# u*v
		sim = _similarity(anchor, sample) / self.tau
		# 分母 u*v正样本+u*v其余负样本
		exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
		# log(exp(uv)/exp(sum(u*v)))=u*v-log(exp(sum(u*v)))
		log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
		# log_prob每一个点都是一个节点和正样本及其所有负样本的loss
		# log_prob * pos_mask是将所有节点取一个正样本和其所有负样本
		loss = log_prob * pos_mask
		# 将所有loss求和，然后取平均
		loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
		return -loss.mean()


class DebiasedInfoNCE(Loss):
	def __init__(self, tau, tau_plus=0.1):
		super(DebiasedInfoNCE, self).__init__()
		self.tau = tau
		self.tau_plus = tau_plus

	def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
		num_neg = neg_mask.int().sum()
		sim = _similarity(anchor, sample) / self.tau
		exp_sim = torch.exp(sim)

		pos_sum = (exp_sim * pos_mask).sum(dim=1)
		pos = pos_sum / pos_mask.int().sum(dim=1)
		neg_sum = (exp_sim * neg_mask).sum(dim=1)
		ng = (-num_neg * self.tau_plus * pos + neg_sum) / (1 - self.tau_plus)
		ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / self.tau))

		log_prob = sim - torch.log((pos + ng).sum(dim=1, keepdim=True))
		loss = log_prob * pos_mask
		loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
		return loss.mean()


class HardnessInfoNCE(Loss):
	def __init__(self, tau, tau_plus=0.1, beta=1.0):
		super(HardnessInfoNCE, self).__init__()
		self.tau = tau
		self.tau_plus = tau_plus
		self.beta = beta

	def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
		# 从这改
		# 负样本数目
		# tau是一个个性化参数，不用理会
		num_neg = neg_mask.int().sum()
		# 每一个位置就是两节点间的相似度
		sim = _similarity(anchor, sample) / self.tau
		# 取exp
		exp_sim = torch.exp(sim)

		# pos是每个样本和节点本身，以及增强图对应节点的相似性
		pos = (exp_sim * pos_mask).sum(dim=1) / pos_mask.int().sum(dim=1)
		# imp是负样本的得分，及每个样本和除两个正样本外所有负样本的得分

		imp = torch.exp(self.beta * (sim * neg_mask))
		sorted, indices = torch.sort(imp, dim=1, descending=True)
		imp = imp / imp.sum(dim=1, keepdim=True)
		# 加权后的负样本
		reweight_neg = (imp * (exp_sim * neg_mask)).sum(dim=1)
		# reweight_neg = imp.sum(dim=1) / imp.mean(dim=1)
		# ng = (-num_neg * self.tau_plus * pos + reweight_neg) / (1 - self.tau_plus)
		# 负样本得分缩放
		# ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / self.tau))

		log_prob = sim - torch.log((pos + reweight_neg)).view(sim.size(0), -1)
		loss = log_prob * pos_mask
		loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
		return -loss.mean()


class HardMixingLoss(torch.nn.Module):
	def __init__(self, projection, tau=0.2):
		super(HardMixingLoss, self).__init__()
		self.projection = projection
		self.tau = tau

	@staticmethod
	def tensor_similarity(z1, z2):
		z1 = F.normalize(z1, dim=-1)  # [N, d]
		z2 = F.normalize(z2, dim=-1)  # [N, s, d]
		return torch.bmm(z2, z1.unsqueeze(dim=-1)).squeeze()

	def forward(self, z1: torch.Tensor, z2: torch.Tensor, threshold=0.1, s=150, mixup=0.2, *args, **kwargs):
		f = lambda x: torch.exp(x / self.tau)
		num_samples = z1.shape[0]
		device = z1.device

		threshold = int(num_samples * threshold)

		refl1 = _similarity(z1, z1).diag()
		refl2 = _similarity(z2, z2).diag()
		pos_similarity = f(_similarity(z1, z2))
		neg_similarity1 = torch.cat([_similarity(z1, z1), _similarity(z1, z2)], dim=1)  # [n, 2n]
		neg_similarity2 = torch.cat([_similarity(z2, z1), _similarity(z2, z2)], dim=1)
		neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
		neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)
		neg_similarity1 = f(neg_similarity1)
		neg_similarity2 = f(neg_similarity2)
		z_pool = torch.cat([z1, z2], dim=0)
		hard_samples1 = z_pool[indices1[:, :threshold]]  # [N, k, d]
		hard_samples2 = z_pool[indices2[:, :threshold]]
		hard_sample_idx1 = torch.randint(hard_samples1.shape[1], size=[num_samples, 2 * s]).to(device)  # [N, 2 * s]
		hard_sample_idx2 = torch.randint(hard_samples2.shape[1], size=[num_samples, 2 * s]).to(device)
		hard_sample_draw1 = hard_samples1[
			torch.arange(num_samples).unsqueeze(-1), hard_sample_idx1]  # [N, 2 * s, d]
		hard_sample_draw2 = hard_samples2[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx2]
		hard_sample_mixing1 = mixup * hard_sample_draw1[:, :s, :] + (1 - mixup) * hard_sample_draw1[:, s:, :]
		hard_sample_mixing2 = mixup * hard_sample_draw2[:, :s, :] + (1 - mixup) * hard_sample_draw2[:, s:, :]

		h_m1 = self.projection(hard_sample_mixing1)
		h_m2 = self.projection(hard_sample_mixing2)

		neg_m1 = f(self.tensor_similarity(z1, h_m1)).sum(dim=1)
		neg_m2 = f(self.tensor_similarity(z2, h_m2)).sum(dim=1)
		pos = pos_similarity.diag()
		neg1 = neg_similarity1.sum(dim=1)
		neg2 = neg_similarity2.sum(dim=1)
		loss1 = -torch.log(pos / (neg1 + neg_m1 - refl1))
		loss2 = -torch.log(pos / (neg2 + neg_m2 - refl2))
		loss = (loss1 + loss2) * 0.5
		loss = loss.mean()
		return loss

	def computer(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, z4: torch.Tensor, threshold=0.1, s=150,
				 mixup=0.2, *args, **kwargs):
		f = lambda x: torch.exp(x / self.tau)
		num_samples = z1.shape[0]
		device = z1.device

		threshold = int(num_samples * threshold)

		refl1 = _similarity(z1, z1).diag()
		refl2 = _similarity(z2, z2).diag()
		pos_similarity = f(_similarity(z1, z2))
		neg_similarity1 = torch.cat([_similarity(z1, z1), _similarity(z1, z2)], dim=1)  # [n, 2n]
		neg_similarity2 = torch.cat([_similarity(z2, z1), _similarity(z2, z2)], dim=1)
		neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
		neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)
		neg_similarity1 = f(neg_similarity1)
		neg_similarity2 = f(neg_similarity2)
		z_pool = torch.cat([z1, z2], dim=0)
		hard_samples1 = z_pool[indices1[:, :threshold]]  # [N, k, d]
		hard_samples2 = z_pool[indices2[:, :threshold]]
		hard_sample_idx1 = torch.randint(hard_samples1.shape[1], size=[num_samples, 2 * s]).to(device)  # [N, 2 * s]
		hard_sample_idx2 = torch.randint(hard_samples2.shape[1], size=[num_samples, 2 * s]).to(device)
		hard_sample_draw1 = hard_samples1[
			torch.arange(num_samples).unsqueeze(-1), hard_sample_idx1]  # [N, 2 * s, d]
		hard_sample_draw2 = hard_samples2[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx2]
		hard_sample_mixing1 = mixup * hard_sample_draw1[:, :s, :] + (1 - mixup) * hard_sample_draw1[:, s:, :]
		hard_sample_mixing2 = mixup * hard_sample_draw2[:, :s, :] + (1 - mixup) * hard_sample_draw2[:, s:, :]

		h_m1 = self.projection(hard_sample_mixing1)
		h_m2 = self.projection(hard_sample_mixing2)

		neg_m1 = f(self.tensor_similarity(z1, h_m1)).sum(dim=1)
		neg_m2 = f(self.tensor_similarity(z2, h_m2)).sum(dim=1)
		pos = pos_similarity.diag()
		neg1 = neg_similarity1.sum(dim=1)
		neg2 = neg_similarity2.sum(dim=1)
		loss1 = -torch.log(pos / (neg1 + neg_m1 - refl1))
		loss2 = -torch.log(pos / (neg2 + neg_m2 - refl2))
		loss = (loss1 + loss2) * 0.5
		loss = loss.mean()
		return loss


class RingLoss(torch.nn.Module):
	# 该算法是一个负采样算法，其思想就是在负采样时选择相似度在[阈值1，阈值2]间的负样本，阈值可以进行退火，即根据训练的轮次进行改变
	def __init__(self, tau):
		super(RingLoss, self).__init__()
		self.tau = tau

	def forward(self, h1: torch.Tensor, h2: torch.Tensor, threshold=0.1, *args, **kwargs):
		f = lambda x: torch.exp(x / self.tau)
		num_samples = h1.shape[0]
		device = h1.device
		threshold = int(num_samples * threshold)

		# false_neg_mask = torch.zeros((num_samples, 2 * num_samples), dtype=torch.int).to(device)
		# for i in range(num_samples):
		#     false_neg_mask[i] = (y == y[i]).repeat(2)

		pos_sim = f(_similarity(h1, h2))
		neg_sim1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)  # [n, 2n]
		neg_sim2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)
		neg_sim1, indices1 = torch.sort(neg_sim1, descending=True)
		neg_sim2, indices2 = torch.sort(neg_sim2, descending=True)

		# y_repeated = y.repeat(2)
		# false_neg_cnt = torch.zeros((num_samples)).to(device)
		# for i in range(num_samples):
		#     false_neg_cnt[i] = (y_repeated[indices1[i, threshold:-threshold]] == y[i]).sum()

		neg_sim1 = f(neg_sim1[:, threshold:-threshold])
		neg_sim2 = f(neg_sim2[:, threshold:-threshold])

		pos = pos_sim.diag()
		neg1 = neg_sim1.sum(dim=1)
		neg2 = neg_sim2.sum(dim=1)

		loss1 = -torch.log(pos / neg1)
		loss2 = -torch.log(pos / neg2)

		loss = (loss1 + loss2) * 0.5
		loss = loss.mean()

		return loss


# K是候选集数量
class DynamicLoss(torch.nn.Module):
	# 该算法是一个负采样算法，其思想就是在负采样时选择相似度在[阈值1，阈值2]间的负样本，阈值可以进行退火，即根据训练的轮次进行改变
	def __init__(self, tau, g, K=100):
		super(DynamicLoss, self).__init__()
		self.tau = tau
		self.K = K
		self.g = torch_geometric.utils.to_networkx(g)
		# 边和点的数量
		self.num_of_nodes = self.g.number_of_nodes()
		self.num_of_edges = self.g.number_of_edges()

		# 边和点的集合
		self.edges_raw = self.g.edges()
		self.nodes_raw = self.g.nodes()
		self.embedding = []  # 嵌入向量
		self.neg_nodeset = {}
		# 为每一个node维护一个负采样群
		self.neg_proportion = {}
		# self.neg_nodeset =[]#负采样点集
		self.node_index = {}  # 节点索引
		self.node_index_reversed = {}
		for index, node in enumerate(self.nodes_raw):
			self.node_index[node] = index  # {node：index}
			self.node_index_reversed[index] = node  # {index:node}
		# 将边转化为index形式
		self.node_set = set([i for i in self.node_index.keys()])
		self.node_index_set = set([i for i in self.node_index.values()])
		self.edges = [(self.node_index[u], self.node_index[v]) for u, v in self.edges_raw]
		# 按照节点原本的度
		node_degrees = [val for (node, val) in self.g.degree()]
		self.node_degree_dict = {}
		for (node, val) in self.g.degree():
			# 节点度字典
			self.node_degree_dict[node] = val

		self.node_degree = node_degrees
		node_degrees /= np.sum(node_degrees)
		self.node_degree_distribution11 = node_degrees
		# 按照3/4度进行采样
		node34_degrees1 = np.power(self.node_degree, 3 / 4)
		# 未归一化的度的3、4
		self.node_degree_distribution341 = node34_degrees1
		node34_degrees = node34_degrees1 / np.sum(node34_degrees1)
		self.node_degree_distribution34 = node34_degrees
		for index in self.node_index.values():
			self.neg_proportion[index] = 0
		for node in self.node_index_reversed.keys():
			# 随机初始化避免选择节点本身做负样本
			self.node_index_set.discard(node)
			node_neg_init_list = list(
				np.random.choice(list(self.node_index_set), size=K,
								 replace=False))
			self.neg_nodeset[node] = node_neg_init_list
			self.node_index_set.add(node)

	# 	可以考虑使用增强图进行负采样

	def forward(self, h1: torch.Tensor, h2: torch.Tensor, threshold=0.1, *args, **kwargs):
		f = lambda x: torch.exp(x / self.tau)
		num_samples = h1.shape[0]
		device = torch.device('cuda')
		# 采样数
		threshold = int(num_samples * threshold)
		#
		# threshold = 8

		# 节点间相似度
		pos_sim = f(_similarity(h1, h2))
		# 单图相似度 和 图间相似度
		neg_sim1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)  # [n, 2n]
		#  图间相似度 和 单图相似度
		neg_sim2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)

		# 负采样
		for node in self.node_index_reversed.keys():
			# 随机初始化避免选择节点本身做负样本
			self.node_index_set.discard(node)
			node_neg_init_list = list(
				np.random.choice(list(self.node_index_set), size=self.K,
								 replace=False))
			self.neg_nodeset[node] = node_neg_init_list
			self.node_index_set.add(node)
		neg_cand = [i for i in self.neg_nodeset.values()]
		neg_cand_gpu = torch.LongTensor(neg_cand)
		neg_cand_gpu = neg_cand_gpu.to(device)
		neg_mask = torch.zeros_like(pos_sim)
		# neg_mask.to(device)
		neg_mask_one = torch.ones_like(pos_sim)
		# neg_mask_one.to(device)
		# print(neg_mask.is_cuda)
		# print(neg_mask_one.is_cuda)
		# print(neg_cand_gpu.is_cuda)

		neg_mask.scatter_(1, neg_cand_gpu, neg_mask_one)
		neg_sim = neg_mask * pos_sim
		neg1 = neg_sim.sum(dim=1)
		neg2 = neg_sim.sum(dim=1)
		# 排序
		# 每个节点的负样本排序
		neg_sim1, indices1 = torch.sort(neg_sim1, descending=True)
		# 排序
		neg_sim2, indices2 = torch.sort(neg_sim2, descending=True)

		neg_sim1 = f(neg_sim1[:, threshold:-threshold])
		neg_sim2 = f(neg_sim2[:, threshold:-threshold])
		# 取对应排序的负样本
		# neg_sim1 = f(neg_sim1[:, :threshold])
		# neg_sim2 = f(neg_sim2[:, :threshold])

		# 取对角线,也就是对应节点的相似度,类似于 pos_sim* E
		pos = pos_sim.diag()
		# 负样本求和
		# neg1 = neg_sim1.sum(dim=1)
		# neg2 = neg_sim2.sum(dim=1)
		#
		loss1 = -torch.log(pos / (pos + neg1))
		loss2 = -torch.log(pos / (pos + neg2))

		loss = (loss1 + loss2) * 0.5
		loss = loss.mean()

		return loss


if __name__ == '__main__':
	x = torch.tensor([[1., 1., 1.], [1., 1., 1.]])
	y = torch.tensor([[2.5, 2.5, 2.5], [4., 4., 3.]])
	z = _similarity(x, y)
	print(z)
	nce = InfoNCE(tau=0.2)
	pos_mask = torch.eye(2, dtype=torch.float32)
	neg_mask = 1. - pos_mask

	loss = nce.compute(x, y, pos_mask, neg_mask)
