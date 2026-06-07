import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# 1. Carregamento dos dados
caminho_arquivo = r # ---> Coloque o caminho do arquivo aqui.

# A métrica 1 
m1 = pd.read_excel(caminho_arquivo, sheet_name='metrica_1', skiprows=2)

# A métrica 2 
m2 = pd.read_excel(caminho_arquivo, sheet_name='metrica_2')

# 2. Feature Engineering (Agregação da Série Temporal)
m1_feat = pd.DataFrame({'country': m1['Código dos Países']})
m1_feat['dep_mean'] = m1.loc[:, 2000 : 2024].mean(axis=1)
m1_feat['dep_delta'] = m1[2024] - m1[2000]

m2_pivot = m2.pivot(index='country_iso3_code', columns='year', values='balanca_comercial').reset_index()
m2_feat = pd.DataFrame({'country': m2_pivot['country_iso3_code']})
m2_feat['bal_mean'] = m2_pivot.loc[:, 2000:2024].mean(axis=1)
m2_feat['bal_delta'] = m2_pivot[2024] - m2_pivot[2000]

# Merge final
df_cluster = pd.merge(m1_feat, m2_feat, on='country')

# Tratamento de Nulos 
df_cluster = df_cluster.dropna()

# 3. Pré-Processamento (Padronização)
features = ['dep_mean', 'dep_delta', 'bal_mean', 'bal_delta']
scaler = StandardScaler()
X = scaler.fit_transform(df_cluster[features])

# 4. Avaliação do Número de Clusters (k)
inertias = []
silhouettes = []
K = range(2, 8)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# --- Cálculo Elbow Method ---
p0 = np.array([K[0], inertias[0]])
pn = np.array([K[-1], inertias[-1]])

distancias = []
for i in range(len(K)):
    p = np.array([K[i], inertias[i]])
    dist = np.linalg.norm(np.cross(pn - p0, p0 - p)) / np.linalg.norm(pn - p0)
    distancias.append(dist)

k_otimo = K[np.argmax(distancias)]
print(f"O método escolheu automaticamente k={k_otimo}")
# -------------------------------------------------------------

# Plot 1: Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K, inertias, marker='o', linestyle='--', color='#2ca02c')
plt.title('Elbow Method')
plt.xlabel('Cluster numbers (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('cotovelo.png', bbox_inches='tight')
plt.close()

# 5. Execução do K-Means Final 
kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans_final.fit_predict(X)

# Recuperar Centróides reais
centroids_unscaled = pd.DataFrame(
    scaler.inverse_transform(kmeans_final.cluster_centers_), 
    columns=features
)
print("Centróides não padronizados:")
print(centroids_unscaled)

# Plot 4: Gráfico de Dispersão das Variações (Deltas)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cluster, x='dep_delta', y='bal_delta', hue='cluster', palette='viridis', s=120)
for i in range(df_cluster.shape[0]):
    plt.text(df_cluster['dep_delta'].iloc[i] + 0.005, 
             df_cluster['bal_delta'].iloc[i], 
             df_cluster['country'].iloc[i], 
             fontsize=9)
plt.title('Dependence variantion vs Trade Balance variation (2000-2024)')
plt.xlabel('Dependency Delta')
plt.ylabel('Trade Balance Delta')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_clusters.png', bbox_inches='tight')
plt.close()

# Salvar o painel de resultados final
df_cluster.to_csv('clusters_g20_china.csv', index=False)