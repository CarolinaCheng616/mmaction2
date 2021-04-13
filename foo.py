import torch


def forward(x, n_data, eps):
    batchsize = x.shape[0]
    m = x.shape[1] - 1

    Pn = 1 / float(n_data)
    import pdb

    pdb.set_trace()
    P_pos = x.select(1, 0)
    log_D1 = torch.div(
        P_pos, P_pos.add(m * Pn + eps)
    ).log_()  # log[1 - const/exp^(f(x)g(y)/T)]

    P_neg = x.narrow(1, 1, m)
    log_D0 = torch.div(
        P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)
    ).log_()  # [batch, bank]
    # log[1 - exp/(exp+const)]

    loss_pos = -log_D1.sum(0) / batchsize
    loss_neg = -log_D0.view(-1, 1).sum(0) / batchsize


if __name__ == "__main__":
    batch = 5
    bank = 10
    feature = 3
    n_data = 100
    eps = 1e-5
    x = torch.tensor((batch, bank + 1, feature))
    forward(x, n_data, eps)
