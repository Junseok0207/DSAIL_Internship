import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1. # self_loop에 추가하는 degree(2면 self loop의 degree를 2로)

    if isinstance(edge_index, SparseTensor):
        pass

    # Adjacency Matrix를 쓰는 방식이 아니라 edge_index와 edge_weight를 사용한다
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size[1], )), dtype=dtype, device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_wieght, fill_value, num_nodes)
            assert tmp_edge_weight is None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_idnex[1]
        deg = torch.scatter_add(edge_weight, col, dim=0, dim_size=num_nodes) # 각각의 node마다 degree더해줌
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] 

class GCNConv(MessagePassing):
    """
    Args:
        in_channels (int)         : input의 dimension
        out_channels (int)        : output의 dimension
        improved (bool, optional) : 모르겠음
        cached (bool, optional)   :
        add_self_loops (bool, optional) : self_loop를 쓸건지
        normalize
        bias
        **kwargs (optional)             : torch.geometirc.nn.conv.MessagePassing의 추가적인 argument 사용
    """
    def __init__(self, in_channels: int, out_channels: int, improved: bool = False, cached: bool = False,
                add_self_loop: bool = True, normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loop
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_paramter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None`
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize: # normalize(논문에서 A틸다 만드는 과정)
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weigt = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                                                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weigt)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                pass

        x= x @ self.wieght # 논문에서 XW

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None) # 논문에서 A와 XW를 곱하는 과정

        if self.bias is not None:
            out += self.bias
        
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1,1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
        
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


    

