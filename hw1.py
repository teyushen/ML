import torch

(A)
w1true = torch.randn(1)
print(w1true)

(B)

N = 50
D = 3
# makes a D-dim vector
mu = torch.zeros(D)
c = torch.rand(1)
cov = (c * torch.ones((D, D))).float()

idx = [i for i in range(D)]
cov[idx, idx] = 1

px = torch.distributions.MultivariateNormal(loc=mu,covariance_matrix=cov)

# Because the question say draw	ing N samples of N(0,1) noise epsilon
epsilon = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1])) 

y = w1true * px.sample([N])[:,0] + epsilon.sample([N])[:,0]
print(y.shape)


(C)
## 
sa = px.sample([N])
## w1 hat all feature
torch.t(sa).matmul(sa).inverse().matmul(torch.t(sa)).matmul(y)

## I don;t know how to do the w1, because it's a vector not a matrix
