import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import torch
from torch.utils.data import DataLoader, Dataset
from autoformer_model import AutoFormer

df = pd.read_csv('gene_expression_over_time.csv')

genes = df['Gene'].values
print/("Number of unique genes:", len(genes))
expressions = df.drop(columns=['Gene']).values.astype(np.float32)

genes_train, genes_test, expr_train, expr_test = train_test_split(
    genes, expressions, test_size=0.2, random_state=42
)
print("Number of training genes:", len(genes_train))
print("Number of testing genes:", len(genes_test))

class GeneExpressionDataset(Dataset):
    def __init__(self, data):
        self.data = data  # shape: (num_samples, time_points)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx, :-1]  # input time series except last time point
        y = self.data[idx, 1:]   # target is next time points (shifted by 1)
        return torch.tensor(x), torch.tensor(y)

train_dataset = GeneExpressionDataset(expr_train)
test_dataset = GeneExpressionDataset(expr_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

time_points = expr_train.shape[1] - 1  # input length
model = AutoFormer(input_len=time_points, output_len=time_points)  # customize as needed
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(20):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(x_batch)  # model output shape should match y_batch
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} done. Loss: {loss.item():.4f}")


model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch)
        all_preds.append(pred.numpy())
        all_targets.append(y_batch.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Calculate Pearson and Spearman per gene (per sample)
pearson_list = []
spearman_list = []

for i in range(all_targets.shape[0]):
    p, _ = pearsonr(all_targets[i], all_preds[i])
    s, _ = spearmanr(all_targets[i], all_preds[i])
    pearson_list.append(p)
    spearman_list.append(s)

print(f"Mean Pearson correlation: {np.mean(pearson_list):.4f}")
print(f"Mean Spearman correlation: {np.mean(spearman_list):.4f}")