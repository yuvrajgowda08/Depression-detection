import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "roberta-base"
MAX_LEN = 256
HIDDEN = 256

df_tr = pd.read_csv("data/splits/train.csv")
df_va = pd.read_csv("data/splits/val.csv")
df_te = pd.read_csv("data/splits/test.csv")

tokenizer = AutoTokenizer.from_pretrained(ENCODER)
encoder = AutoModel.from_pretrained(ENCODER).to(DEVICE)
encoder.eval()

@torch.no_grad()
def embed_texts(texts, batch_size=16):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tok = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        out = encoder(**tok)
        cls = out.last_hidden_state[:,0,:].detach().cpu().numpy()
        embs.append(cls)
    return np.vstack(embs)

def make_sequences(df):
    df = df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values(["user_id","window_start"])
    users = []
    seqs = []
    labels = []
    for uid, g in df.groupby("user_id"):
        texts = g["text"].tolist()
        X = embed_texts(texts)
        users.append(uid)
        seqs.append(X)
        labels.append(int(g["label"].iloc[0]))
    return users, seqs, np.array(labels, dtype=np.int64)

class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

def collate(batch):
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    dim = seqs[0].shape[1]
    X = np.zeros((len(seqs), maxlen, dim), dtype=np.float32)
    mask = np.zeros((len(seqs), maxlen), dtype=np.float32)
    for i,s in enumerate(seqs):
        X[i,:s.shape[0],:] = s
        mask[i,:s.shape[0]] = 1.0
    return torch.tensor(X), torch.tensor(mask), torch.tensor(labels)

class LSTMHead(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x, mask):
        # pack by lengths
        lengths = mask.sum(1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out,_ = self.lstm(packed)
        out,_ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # take last valid step
        idx = (lengths-1).view(-1,1,1).expand(-1,1,out.size(-1))
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last)

def train_epoch(model, dl, opt):
    model.train()
    ce = nn.CrossEntropyLoss()
    total=0
    for X,mask,y in dl:
        X,mask,y = X.to(DEVICE),mask.to(DEVICE),y.to(DEVICE)
        opt.zero_grad()
        logits = model(X,mask)
        loss = ce(logits,y)
        loss.backward()
        opt.step()
        total += loss.item()*len(y)
    return total/len(dl.dataset)

@torch.no_grad()
def eval_f1(model, dl):
    model.eval()
    ys=[]
    ps=[]
    for X,mask,y in dl:
        X,mask = X.to(DEVICE),mask.to(DEVICE)
        logits = model(X,mask).cpu().numpy()
        pred = logits.argmax(1)
        ys.extend(y.numpy().tolist())
        ps.extend(pred.tolist())
    from sklearn.metrics import f1_score
    return f1_score(ys, ps)

_, tr_seqs, tr_y = make_sequences(df_tr)
_, va_seqs, va_y = make_sequences(df_va)
_, te_seqs, te_y = make_sequences(df_te)

tr_dl = DataLoader(SeqDataset(tr_seqs,tr_y), batch_size=8, shuffle=True, collate_fn=collate)
va_dl = DataLoader(SeqDataset(va_seqs,va_y), batch_size=8, shuffle=False, collate_fn=collate)
te_dl = DataLoader(SeqDataset(te_seqs,te_y), batch_size=8, shuffle=False, collate_fn=collate)

model = LSTMHead(in_dim=tr_seqs[0].shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

best=0
for epoch in range(1,6):
    loss = train_epoch(model, tr_dl, opt)
    f1v = eval_f1(model, va_dl)
    print(f"epoch={epoch} loss={loss:.4f} val_f1={f1v:.4f}")
    best = max(best, f1v)

f1t = eval_f1(model, te_dl)
print("TEST F1:", f1t)
torch.save(model.state_dict(), "outputs/lstm_earlyrisk.pt")
print("Saved outputs/lstm_earlyrisk.pt")
