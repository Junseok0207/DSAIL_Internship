import node2vec
import torch
import argparse
from torch_geometric.datasets import Planetoid


def main():

    parser = argparse.ArgumentParser
    parser.add_argument("-emb_dim", "--embedding_dim", type=int, defalut=128)
    parser.add_argument("-w_len", "--walk_length", type=int, defalut=20)
    parser.add_argument("-c_size", "--context_size", type=int, defalut=10)
    parser.add_argument("-w_per_node", "--walks_per_node", type=int, defalut=10)
    parser.add_argument("-n_neg_sam", "--num_negative_samples", type=int, defalut=1)
    parser.add_argument("-p", "--p", type=int, defalut=1)
    parser.add_argument("-q", "--q", type=int, defalut=1)

    args = parser.parse_args()

    # load dataset
    dataset = Planetoid(root='tmp/Cora', name='Cora')
    data = dataset[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = node2vec.Node2Vec(data.edge_index, embedding_dim=args.embedding_dim, walk_length=args.walk_length, 
                            context_size=args.context_size, walks_per_node=args.walk_length,
                            num_negative_samples=args.num_negative_samples, p=args.p, q=args.q, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
        return acc

    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

if __name__ == "__main__":
    main()



