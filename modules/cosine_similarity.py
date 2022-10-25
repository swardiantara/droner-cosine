import numpy as np
import torch
from torch import linalg as LinearAlgebra
from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
batch_size = 2
seq_len = 4
n_head = 14
d_k = 8
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)
# print("Sebelum Q:", Q)
# print("Sebelum K:", K.size())
# A=np.array([[2,2,3],[1,0,4],[6,9,7]])
# B=np.array([[1,5,2],[6,6,4],[1,10,7],[5,8,2],[3,0,6]])
# A=np.random.rand(1,2,9,128)
# B=np.random.rand(1,2,9,128)
# def csm(A,B):
#     num=np.dot(A, np.transpose(B, (0,1,3,2)))
#     p1=np.sqrt(np.sum(A**2,axis=3))[:,np.newaxis]
#     p2=np.sqrt(np.sum(B**2,axis=3))[np.newaxis,:]
#     return num/(p1*p2)
# print(csm(A,B))
# A = torch.from_numpy(A)
# B= torch.from_numpy(B)
norm_q = LinearAlgebra.vector_norm(Q, dim=-1)[:, :, None]
norm_k = LinearAlgebra.vector_norm(K, dim=-1)[:, :, None]
print(norm_q)
# print(norm_k)
a_norm = Q / norm_q
b_norm = K / norm_k
# b_norm = b_norm.transpose(1, 2)
# b_norm = b_norm.permute(0, 2, 1)
# print("Sesudah Q:", a_norm)
# print("Sesudah K:", b_norm.size())
# res = torch.matmul(a_norm, b_norm)
cosine = a_norm.bmm(b_norm.transpose(1, 2))
dot_product = Q.bmm(K.transpose(1, 2))
skelarn_cosine = CosineSimilarity(Q, K.transpose(1, 2))
print("Cosine Attention :", cosine)
print("Dot Attention :", dot_product)
print("Dot Attention :", skelarn_cosine)
weighted2 = cosine.bmm(V)
weighted = dot_product.bmm(V)
# weighted = torch.matmul(res, V)
# print("Cosine :", weighted2)
# print("Dot Weighted :", weighted)

#  0.9978 -0.9986 -0.9985
# -0.8629  0.9172  0.9172