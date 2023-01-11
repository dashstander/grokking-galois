import copy
import einops
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm.auto as tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
import wandb


def make_dataset(p: int, device):
    # For p**4 elements add a d vector with l -> (i j k l)
    a_vector = einops.repeat(torch.arange(p), "i -> (i j k)", j=p, k=p)
    b_vector = einops.repeat(torch.arange(p), "j -> (i j k)", i=p, k=p)
    c_vector = einops.repeat(torch.arange(p), "k -> (i j k)", i=p, j=p)
    #d_vector = einops.repeat(torch.arange(p), "l -> (i j k l)", i=p, j=p, k=p) 
    equals_vector = einops.repeat(torch.tensor(p), " -> (i j k)", i=p, j=p, k=p)
    star_vector = einops.repeat(torch.tensor(p+1), " -> (i j k)", i=p, j=p, k=p)
    plus_vector = einops.repeat(torch.tensor(p+2), " -> (i j k)", i=p, j=p, k=p)
    #caret_vector = einops.repeat(torch.tensor(p+3), " -> (i j k l)", i=p, j=p, k=p, l=p)
    #lparen = einops.repeat(torch.tensor(p+4), " -> (i j k l)", i=p, j=p, k=p, l=p)
    #rparen = einops.repeat(torch.tensor(p+5), " -> (i j k l)", i=p, j=p, k=p, l=p)
    dataset = torch.stack([
        a_vector,
        star_vector,
        b_vector,
        plus_vector,
        c_vector,
        equals_vector], dim=1).to(device)

    labels = ((dataset[:, 0] * dataset[:, 2] + dataset[:, 4])) % p
    return dataset, labels


def get_dataloaders(p: int, frac_train: float, batch_size: int, device):
    dataset, labels = make_dataset(p, device)
    indices = torch.randperm(p**4)
    cutoff = int((p**3) * frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]
    train_data = TensorDataset(dataset[train_indices], labels[train_indices])
    test_data = TensorDataset(dataset[test_indices], labels[test_indices])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def loss_fn(logits, labels):
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for batch, labels in dataloader:
        logits = model(batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        total_loss += loss
    return total_loss


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for batch, labels in dataloader:
        logits = model(batch)
        loss = loss_fn(logits, labels)
        total_loss += loss
    return total_loss


def train(model, optimizer, train_dataloader, test_dataloader, checkpoint_every, num_epochs, grok_threshold):
    train_losses = []
    test_losses = []
    model_checkpoints = []
    checkpoint_epochs = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss = train_forward(model, train_dataloader)
        np_train = train_loss.item()
        train_losses.append(np_train)

        optimizer.step()
        optimizer.zero_grad()

        with torch.inference_mode():
            test_loss = test_forward(model, test_dataloader)
            np_test = test_loss.item()
            test_losses.append(np_test)
        
        wandb.log({'loss/train': np_train, 'loss/test': np_test})
        
        if (epoch % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
            model_checkpoints.append(copy.deepcopy(model.state_dict()))
            print(f"Epoch {epoch} Train Loss {np_train} Test Loss {np_test}")
        if test_loss.item() <= grok_threshold:
            break


def main():
    p = 53
    frac_train = 0.5
    lr = 1e-3
    wd = 1. 
    betas = (0.9, 0.98)
    num_epochs = 200_000
    grok_threshold = 0.01
    checkpoint_every = 100
    batch_size = 2 ** 16
    device = 'cuda'
    seed = 999

    cfg = HookedTransformerConfig(
        n_layers = 1,
        n_heads = 4,
        d_model = 256,
        d_head = 64,
        d_mlp = 1024,
        act_fn = "relu",
        normalization_type=None,
        d_vocab=p+3,
        d_vocab_out=p,
        n_ctx=6,
        init_weights=True,
        device=device,
        seed = seed,
    )
    model = HookedTransformer(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=betas
    )

    wandb.init(
        entity="dstander",
        project="grokking_galois",
        group="x*y+z",
        config=cfg.to_dict()
    )

    wandb.watch(model, log_freq=100)

    train_data, test_data = get_dataloaders(p, frac_train, batch_size, device)

    train(
        model,
        optimizer,
        train_data,
        test_data,
        checkpoint_every,
        num_epochs,
        grok_threshold
    )


if __name__ == '__main__':
    main()

