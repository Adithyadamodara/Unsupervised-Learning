
# p1.ipynb
import pandas as pd

data = pd.read_csv("reviews1.csv")

data.info()
!python -m spacy download en_core_web_sm
# Takes ~2 mins to run

import re
import spacy


# Spacy - trained pipeline package for
# en - english
# core - vocabulary, syntax, entities
# web - trained on web data (blogs, news and comments)
# sm - small size for speed, uses context sensitive tensors 
nlp = spacy.load("en_core_web_sm")


data = data[['content', 'score', 'at', 'appId']].copy()


# Dropping duplicates to avoid model memorization
data.drop_duplicates(subset=['content'], inplace = True)


# Removing any review with length < 5, to reduce noise and keep meaningful data
# Remaining 10330 rows with meaningful, removed 1477 
data = data[data['content'].str.split().str.len() >= 5].reset_index(drop=True)


def clean_text(text):
    text = text.lower()

    # Removing markdown email links 
    text = re.sub(r'\[.*?\]\(mailto:.?\)', '', text)

    # Removing plain emails
    text = re.sub(r'\S+@\S+', '', text)

    # Removing URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Removing HTML tags 
    text = re.sub(r'<.*?>', '', text)

    # Normalizing emojis as punctutaion (space here)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Removing special characters (keeping apostrophes for contractions, eg: "don't")
    text = re.sub(r"[^a-z0-9\s']", ' ', text)

    # Normalizing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# Using apply can make it slow for millions of rows, but okay for thousands  
data['content_clean'] = data['content'].apply(clean_text)


APP_NAME_TOKENS = {
    'anydo', 'todoist', 'task', 'habitica','forestapp' ,'forestapp', 'habitbull',
    'todos', 'timetune', 'bizcal', 'planner', 'calclock', 'habitnow', 'liferpgtasks', 'artfulagenda'
}

# Removing stopwords
DOMAIN_STOPWORDS = {
    'app', 'apps', 'please', 'thanks', 'thank', 'hi', 'hello', 'use',
    'used', 'using', 'get' , 'got', 'would', 'could', 'update', 'updated', 
    'version', 'phone', 'day', 'time', 'list', 'tap' ,'open' ,'show'
}

DOMAIN_STOPWORDS = DOMAIN_STOPWORDS | APP_NAME_TOKENS


def lemmatize(text):
    doc = nlp(text)

    # lemma_ reduces words to dictionary from; 'running','ran','runs' - 'run', 'better' - 'good'
    # .is_stop : identifies common words - this, is, at....
    # .is_alpha : identifies standard words and removes numbers, puncutations, emojis, mixed strings... 
    tokens =[
        token.lemma_ for token in doc  
        if not token.is_stop
        and token.is_alpha
        and len(token.text) > 2 # ignores very short tokens 
        and token.lemma_ not in DOMAIN_STOPWORDS 
    ] 

    return ' '.join(tokens)


data['content_processed'] = data['content_clean'].apply(lemmatize)
# Filtering length after lemmatization
# Since about ~800 tokens each are of length 3 and 4 its removed to retain meaning
data = data[data['content_processed'].str.split().str.len() >= 5].reset_index(drop=True)

print(f"Final dataset size: {len(data)}")
print(data[['content', 'content_processed']].head(10))
# Checking token length distribution post-processing
lengths = data['content_processed'].str.split().str.len()
print(lengths.describe())
print(lengths.value_counts().sort_index().head(10))
# Since there are about 263 tokens(max), it needs to be capped, sentence transformers have 256-512 token limit,
# Extremely long reviews get truncated at the tail and lose meaning.

data['content_processed'] = data['content_processed'].apply(
    lambda x: ' '.join(x.split()[:100])
)

lengths = data['content_processed'].str.split().str.len()
print(f"Final row count: {len(data)}")
print(f"Min tokens: {lengths.min()}")
print(f"Reviews with < 5 tokens: {(lengths < 5).sum()}")
print(f"Max tokens: {lengths.max()}")
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Using GPU for quicker execution
# ~37s using GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLm-L6-v2', device = device)

# using cleaned test, instead of preprocessed, as preprocessed text looses too much meaning when passed to sentence transformers    
embeddings = model.encode(
    data['content_clean'].tolist(), 
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Embeddings shape: {embeddings.shape}")

# Saving to avoid recomputing
np.save('embeddings.npy', embeddings)
print("Saved!")
# Verification using cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

# Picking 3 pairs manually from data
# Pair A: two reviews that should be SIMIIAR 
# Pair B: two reviews that should be DIFFERENT

# Reviews about price 
pricing_idx = data[data['content_clean'].str.contains('costly|expensive|price|rupee|refund')].index[:2].tolist()

# Reviews about crashes/bugs
crash_idx = data[data['content_clean'].str.contains('crash|freeze|bug|error')].index[:2].tolist()



# Comparison
print("===SIMILAR PAIR (both pricing)===")
print("A:", data['content_clean'].iloc[pricing_idx[0]])
print("B:", data['content_clean'].iloc[pricing_idx[1]])
sim = cosine_similarity([embeddings[pricing_idx[0]]], [embeddings[pricing_idx[1]]])[0][0]
print(f"Cosine similarity: {sim:.4f}")

print("\n=== DIFFERENT PAIR (pricing vs crash) ===")
print("A:", data['content_clean'].iloc[pricing_idx[0]])
print("B:", data['content_clean'].iloc[crash_idx[0]])
sim2 = cosine_similarity([embeddings[pricing_idx[0]]], [embeddings[crash_idx[0]]])[0][0]
print(f"Cosine similarity: {sim2:.4f}")

print("\n=== RANDOM PAIR (should be somewhere in between) ===")
sim3 = cosine_similarity([embeddings[0]], [embeddings[100]])[0][0]
print(f"Cosine similarity: {sim3:.4f}")
# Comparing more cleaned sentences. 
print(data[data['content_clean'].str.contains('sync')][['content_clean']].head(5))
print(data[data['content_clean'].str.contains('widget')][['content_clean']].head(5))
print(data[data['content_clean'].str.contains('notification')][['content_clean']].head(5))
sync_idx = [6, 16, 31, 59, 100]
widget_idx = [7, 12, 23, 107, 126]
notification_idx = [9, 11, 20, 49, 57]

print("SYNC vs SYNC:")
s1 = cosine_similarity([embeddings[sync_idx[0]]], [embeddings[sync_idx[1]]])[0][0]
print(f"{s1:.4f}")

print("SYNC vs WIDGET:")
s2 = cosine_similarity([embeddings[sync_idx[0]]],[embeddings[widget_idx[0]]])[0][0]
print(f"{s2:.4f}")

print("SYNC vs CRASH:")
s3 = cosine_similarity([embeddings[sync_idx[0]]],[embeddings[crash_idx[0]]])[0][0]
print(f"{s3:.4f}")

print("WIDGET vs WIDGET:")
s4 = cosine_similarity([embeddings[widget_idx[0]]],[embeddings[widget_idx[1]]])[0][0]
print(f"{s4:.4f}")


# Structural sanity check

print("NaNs: ", np.isnan(embeddings).sum())
print("Infs: ", np.isinf(embeddings).sum())

norms = np.linalg.norm(embeddings, axis=1)

print(f"Norm mean: {norms.mean():.4f}, std: {norms.std():.4f}")

# Loose T-SNE (sample)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Using 2k subsample instead of entire 7k for speed
sample_idx = np.random.choice(len(embeddings), 2000, replace=False)
sample_embeddings = embeddings[sample_idx]
sample_scores = data['score'].iloc[sample_idx].values

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced = tsne.fit_transform(sample_embeddings)

plt.figure(figsize=(10,7))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=sample_scores, cmap='RdYlGn', alpha=0.5, s=10)

plt.colorbar(scatter, label='Review Score')
plt.title('t-SNE of Review Embeddings (Colored by score)')
plt.savefig('tsne_phase2.png', dpi=75)
plt.show()
assert len(data) == embeddings.shape[0]

data.to_csv('reviews_preprocessed.csv', index=False)
print("Reviews saved!")

# p2.ipynb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

embeddings = np.load('embeddings.npy')
X = torch.tensor(embeddings, dtype=torch.float32).to(device)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)   
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        return self.encoder(x)


model = Autoencoder(input_dim=384, latent_dim=32).to(device)
print(model)
print(f"\nTotal paramters: {sum(p.numel() for p in model.parameters()):,}")
# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)
criterion = nn.MSELoss()

EPOCHS = 50
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for (batch,) in loader:
        optimizer.zero_grad()
        recon, z = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss/len(loader)
    loss_history.append(avg_loss)
    scheduler.step(avg_loss)

    if (epoch + 1) % 5 == 0 :
        print(f"Epoch {epoch+1:3d} / {EPOCHS} | Loss: {avg_loss:.6f}")

plt.figure(figsize=(8,4))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('AutoEncoder Training Loss')
plt.grid(True)
plt.savefig('autoencoder_loss.png', dpi=100)
plt.show()
# Extracting latent vectors 
model.eval()
with torch.no_grad():
    X_full = torch.tensor(embeddings, dtype=torch.float32).to(device)
    recon_full, Z = model(X_full)
    Z = Z.cpu().numpy()
    recon_full = recon_full.cpu().numpy()

print(f"Latent space shape: {Z.shape}")
# Expected: (7824, 32)

np.save('latent_vectors.npy', Z)
# Running k-means to verify autoencoder effectiveness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

scaler_emb = StandardScaler()
scaler_z = StandardScaler()

X_scaled = scaler_emb.fit_transform(embeddings)
Z_scaled = scaler_z.fit_transform(Z)

# Fixed k for comparison 
K = 10

print("Running KMeans on raw embeddings....")
km_raw = KMeans(n_clusters=K, random_state=42, n_init=10)
labels_raw = km_raw.fit_predict(X_scaled)
sil_raw = silhouette_score(X_scaled, labels_raw, sample_size=2000)

print("Running Kmeans on latent vectors....")
km_latent = KMeans(n_clusters=K, random_state=42, n_init=10)
labels_latent = km_latent.fit_predict(Z_scaled)
sil_latent = silhouette_score(Z_scaled, labels_latent, sample_size=2000)

print(f"\nSilhouette Score for Raw embeddings (384d): {sil_raw:.4f}")
print(f"\nSilhouette Score for Latent vectors (384d): {sil_latent:.4f}")
# Reconstruction Sanity check
# Check how well the decoder reconstructs the original embeddings
import pandas as pd

data = pd.read_csv('reviews_preprocessed.csv')


recon_errors = np.mean((embeddings - recon_full) ** 2, axis=1)
print(f"Mean reconstruction error: {recon_errors.mean():.6f}")
print(f"Std reconstruction error:  {recon_errors.std():.6f}")
print(f"Max reconstruction error:  {recon_errors.max():.6f}") 

# Flagging outliers
outlier_threshold = recon_errors.mean() + 3 * recon_errors.std()
outliers = np.where(recon_errors > outlier_threshold)[0] 
print(f"\nHigh reconstruction error reviews: {len(outliers)}")
print("Sample outlier reviews:")
for idx in outliers[:3]:
    print(f"  [{idx}] {data['content_clean'].iloc[idx][:100]}")
# T-SNE on latent space

from sklearn.manifold import TSNE

sample_idx = np.random.choice(len(Z), 2000, replace=False)
Z_sample = Z[sample_idx]
scores_sample = data['score'].iloc[sample_idx].values

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
Z_2d = tsne.fit_transform(Z_sample)

plt.figure(figsize=(10,7))
scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=scores_sample, cmap='RdYlGn', alpha=0.5 , s=10)
plt.colorbar(scatter, label='Review Score')
plt.title('t-SNE of Latent Space — Post Autoencoder')
plt.savefig('tsne_phase3_latent.png', dpi=150)
plt.show()
# Saving model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_history': loss_history,
    'final_loss': loss_history[-1]
}, 'autoencoder_checkpoint.pt')

print(f"Model saved. Final loss: {loss_history[-1]:.6f}")
# For loading
"""checkpoint = torch.load('autoencoder_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() """
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 

Z = np.load('latent_vectors.npy')
Z_scaled = StandardScaler().fit_transform(Z)

K_range = range(5,21)
inertias = []
silhouettes = []
db_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=15) 
    labels = km.fit_predict(Z_scaled)

    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Z_scaled, labels, sample_size=2000, random_state=42))
    db_scores.append(davies_bouldin_score(Z_scaled, labels))

    print(f"K={k:2d} | Inertia: {km.inertia_:,.1f} | Silhouette: {silhouettes[-1]:.4f} | DB: {db_scores[-1]:.4f}")
# Plotting all three metric outputs
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('Elbow Curve')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')
axes[0].grid(True)

axes[1].plot(K_range, silhouettes, 'go-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Score (higher = better)')
axes[1].grid(True)

axes[2].plot(K_range, db_scores, 'ro-')
axes[2].set_title('Davies-Bouldin Index')
axes[2].set_xlabel('K')
axes[2].set_ylabel('Score (lower = better)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('k_selection.png', dpi=150)
plt.show()
# Manual check of K
import pandas as pd

data = pd.read_csv('reviews_preprocessed.csv')

k_chosen = 6
km_final = KMeans(n_clusters=k_chosen, random_state=42, n_init=15)
labels_init = km_final.fit_predict(Z_scaled)

print("==== CLUSTER SAMPLE REVIEW ====\n")

for cluster_id in range(k_chosen):
    cluster_mask = labels_init == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    sample = np.random.choice(cluster_indices, min(5, len(cluster_indices)), replace=False)

    print(f"--- Cluster {cluster_id} ({cluster_mask.sum()} reviews) ---")
    for idx in sample:
        print(f"  [{data['score'].iloc[idx]}] - {data['content_clean'].iloc[idx][:120]}")
    print()

# Saving centroids

km_unscaled = KMeans(n_clusters=k_chosen, random_state=42, n_init=15)
km_unscaled.fit(Z)

initial_centroids = km_unscaled.cluster_centers_
print(f"Centroids shape: {initial_centroids.shape}")

np.save('initial_centroids.npy', initial_centroids)
np.save('kmeans_init_labels.npy', km_unscaled.labels_)
print("Centroids saved.")
# Saving Pre-IDEC T-SNE for comparison
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sample_idx = np.random.choice(len(Z), 2000, replace=False)
Z_sample = Z[sample_idx]
labels_sample = km_unscaled.labels_[sample_idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
Z_2d = tsne.fit_transform(Z_sample)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1],
                      c=labels_sample, cmap='tab10',
                      alpha=0.6, s=10)
plt.colorbar(scatter, label='Cluster (K-Means init)')
plt.title('t-SNE, Latent Space Before IDEC (K-Means labels)')
plt.savefig('tsne_before_idec.png', dpi=150)
plt.show()

np.save('tsne_2d_before_idec.npy', Z_2d)
np.save('tsne_sample_idx.npy', sample_idx)
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Loading all necessary files
Z = np.load('latent_vectors.npy')
initial_centroids = np.load('initial_centroids.npy')
embeddings = np.load('embeddings.npy')

# Selecting GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading embeddings into gpu as tensors
X = torch.tensor(embeddings, dtype=torch.float32).to(device)

# Reloading pre-trained autoencoder 
checkpoint = torch.load('autoencoder_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Cluster centroids - learnable paramters
centroids = torch.tensor(
    initial_centroids,
    dtype=torch.float32,
    requires_grad=True,
    device=device
)

# Soft assignment function Q
def compute_q(z, centroids):
    """
    z: (N, latent_dim)
    centroids: (K, latent_dim)
    returns q: (N, K) soft assignments
    """ 
    # Squared distances between each point and each centroid
    # z: (N, 1, D), centroids: (1, K, D) -> diff: (N, K, D)
    diff = z.unsqueeze(1) - centroids.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)

    numerator = 1.0 / (1.0 + dist_sq)
    q = numerator / numerator.sum(dim=1, keepdim=True) 
    return q

def compute_p(q):
    """
    Sharpen q to get target distribution p.
    q: (N, K)
    returns p: (N, K)
    """

    q_sq = q ** 2
    # Normalizing by cluster frequency (column sum)
    freq = q.sum(dim=0, keepdim=True)
    p = (q_sq / freq) / (q_sq / freq).sum(dim=1, keepdim=True)
    return p


# IDEC training setup
GAMMA = 0.1           # Weight on clustering loss vs reconstruction loss
UPDATE_INTERVAL = 50  # How often to recompute P (in batches)
TOL = 0.001           # Convergence threshold: 0.1% assignments changed
EPOCHS = 30
BATCH_SIZE = 256

optimizer_idec = torch.optim.Adam(
    list(model.parameters()) + [centroids], lr=1e-4 # lower than pretraining
)

reconstruction_loss_fn = nn.MSELoss()

dataset = TensorDataset(X, torch.arange(len(X)))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
history = {
    'total_loss': [],
    'recon_loss': [],
    'cluster_loss': [],
    'silhouette': [],
    'pct_changed': []
}


# initalizing P using full dataset before training
model.eval()
with torch.no_grad():
    _, Z_init = model (X)
    Q_global = compute_q(Z_init, centroids)
    P_global = compute_p(Q_global).detach()

prev_labels = Q_global.argmax(dim=1).cpu().numpy()

print("Starting IDEC joint training...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_total = 0
    epoch_recon = 0
    epoch_cluster = 0
    batch_count = 0

    for batch_x, batch_idx in loader:
        optimizer_idec.zero_grad()

        # Forward pass
        recon, z = model(batch_x)

        # Soft assignments for this batch
        q_batch = compute_q(z, centroids)

        # Getting pre-computed P for current batch's indices
        p_batch = P_global[batch_idx]

        # Reconstruction loss
        loss_recon = reconstruction_loss_fn(recon, batch_x)
        
        # KL divergence loss: KL(P || Q)
        # Adding epsilon to avoid log(0)
        eps = 1e-10
        loss_cluster = (p_batch * torch.log((p_batch + eps) / (q_batch + eps))).sum(dim=1).mean()

        # Combined IDEC loss
        loss = loss_recon + GAMMA * loss_cluster

        loss.backward()
        optimizer_idec.step()

        epoch_total += loss.item()
        epoch_recon += loss_recon.item()
        epoch_cluster += loss_cluster.item()
        batch_count += 1

        # END of FOR
    # END of Epoch, recomputing p and checking convergence
    model.eval()
    with torch.no_grad():
        _, Z_current = model(X)
        Q_global = compute_q(Z_current, centroids)
        P_global = compute_p(Q_global).detach()

    current_labels = Q_global.argmax(dim=1).cpu().numpy()

    # Checking % of assignment changes since last epoch
    pct_changed = (current_labels != prev_labels).mean() * 100
    prev_labels = current_labels.copy()

    # Silhouette on current latent space (using subsample for speed)
    sample_idx = np.random.choice(len(Z_current), 2000, replace=False)
    Z_np = Z_current.cpu().numpy()
    sil = silhouette_score(Z_np[sample_idx], current_labels[sample_idx])

    # logging
    n = batch_count 
    history['total_loss'].append(epoch_total / n)
    history['recon_loss'].append(epoch_recon / n)
    history['cluster_loss'].append(epoch_cluster / n)
    history['silhouette'].append(sil)
    history['pct_changed'].append(pct_changed)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Total: {epoch_total/n:.5f} | "
          f"Recon: {epoch_recon/n:.5f} | "
          f"KL: {epoch_cluster/n:.5f} | "
          f"Sil: {sil:.4f} | "
          f"Changed: {pct_changed:.2f}%")

    # Convergence check
    if pct_changed < TOL * 100 and epoch > 5:
        print(f"\nConverged at epoch {epoch+1} - less than {TOL * 100:.1f}% assignments changed.")
        break

print("\n IDEC training complete")
# Final latent vectors and labels 
model.eval()
with torch.no_grad():
    _, Z_final = model(X)
    Q_final = compute_q(Z_final, centroids)

Z_final_np = Z_final.cpu().numpy()
labels_final = Q_final.argmax(dim=1).cpu().numpy()
Q_final_np = Q_final.cpu().numpy()

np.save('latent_vectors_idec.npy', Z_final_np)
np.save('labels_idec.npy', labels_final)
np.save('soft_assignments_idec.npy', Q_final_np)
np.save('centroids_idec.npy', centroids.detach().cpu().numpy())

torch.save({
    'model_state_dict': model.state_dict(),
    'centroids': centroids.detach().cpu(), 
    'history': history
}, 'idec_checkpoint.pt')

print("All saved!")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(history['total_loss'], 'b-o', markersize=3)
axes[0,0].set_title('Total Loss (Recon + γ·KL)')
axes[0,0].set_xlabel('Epoch')
axes[0,0].grid(True)

axes[0,1].plot(history['recon_loss'], 'g-o', markersize=3, label='Reconstruction')
axes[0,1].plot(history['cluster_loss'], 'r-o', markersize=3, label='KL Clustering')
axes[0,1].set_title('Loss Components')
axes[0,1].legend()
axes[0,1].set_xlabel('Epoch')
axes[0,1].grid(True)

axes[1,0].plot(history['silhouette'], 'purple', marker='o', markersize=3)
axes[1,0].axhline(y=0.2032, color='gray', linestyle='--', label='Pre-IDEC baseline')
axes[1,0].set_title('Silhouette Score Over Training')
axes[1,0].set_xlabel('Epoch')
axes[1,0].legend()
axes[1,0].grid(True)

axes[1,1].plot(history['pct_changed'], 'orange', marker='o', markersize=3)
axes[1,1].axhline(y=1.0, color='red', linestyle='--', label='1% threshold')
axes[1,1].set_title('% Assignments Changed Per Epoch')
axes[1,1].set_xlabel('Epoch')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('idec_training_diagnostics.png', dpi=150)
plt.show()
unique, counts = np.unique(labels_final, return_counts=True)
for c, n in zip(unique, counts):
    print(f"Cluster {c}: {n} reviews ({n/len(labels_final)*100:.1f}%)")
# TSNE comparison of pre and post IDEC
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

Z_final_np = np.load('latent_vectors_idec.npy')
labels_final = np.load('labels_idec.npy')

# Reusing same samples for comparison
sample_idx = np.load('tsne_sample_idx.npy')
Z_before_2d = np.load('tsne_2d_before_idec.npy')

# Computing post-IDEC 2d projection on same sample
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
Z_after_2d = tsne.fit_transform(Z_final_np[sample_idx])

labels_sample = labels_final[sample_idx]
colors = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4']

fig, axes = plt.subplots(1, 2, figsize=(16,7))

for ax, Z_2d, title in zip(axes, [Z_before_2d, Z_after_2d],
['Before IDEC (K-Means init)', 'After IDEC (Joint Training)']):
    for k in range(6):
        mask = labels_sample == k
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
            c=colors[k], label=f'Cluster {k}',
            alpha=0.5, s=10)        
    ax.set_title(title, fontsize=13)
    ax.legend(markerscale=2, fontsize=8)
    ax.axis('off')

plt.suptitle('Latent Space: Before VS after IOEC training', fontsize=14)
plt.tight_layout()
plt.savefig('tsne_before_after_idec.png', dpi=150)
plt.show()