import numpy as np
import gudhi as gd
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X = pd.read_excel('/Users/mac/Desktop/infospe/dataset2.xls')
X = X.to_numpy()


# Apply PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)



# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
# # Adding labels and legend
# ax.set_xlabel('caracterstique 1')
# ax.set_ylabel('caracterstique 2')
# ax.set_zlabel('caracterstique 3')
# legend = ax.legend(*scatter.legend_elements(), title="Classes")
# ax.add_artist(legend)

# plt.title("Visualisation 3D de la base de données en PCA")
# plt.show()
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Plot the 3D PCA visualization with color
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Color the points based on their position along the first principal component
colors = X_pca[:, 0]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, cmap='viridis')

ax.set_xlabel('Caractéristique 1')
ax.set_ylabel('Caractéristique 2')
ax.set_zlabel('Caractéristique 3')
ax.set_title('Visualisation de la base de données en dimension 3 par utilisation de la pca')

# Add a colorbar
# cbar = fig.colorbar(sc, shrink=0.5, aspect=5)
# cbar.set_label('Caractéristique 1')

plt.show()


rips_complex = gd.RipsComplex(points=X)


simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)


diag = simplex_tree.persistence()



gd.plot_persistence_diagram(diag)
plt.title("Diagramme de Persistance")
plt.xlabel('Naissance')
plt.ylabel('Mort')
plt.savefig("high_res_plot.png", dpi=300)
plt.show()



gd.plot_persistence_barcode(diag)
plt.title("Code-Barres de Persistance")
plt.show()




anomaly_scores = []
for idx in range(len(X)):
    filtration_value = simplex_tree.filtration(X[idx])
    intervals = simplex_tree.persistence_intervals_in_dimension(0)
    # Calcul du score d'anomalie basé sur la filtration et la persistance
    score = sum(p[1] - p[0] for p in intervals if p[1] != np.inf)
    anomaly_scores.append(score)

# Identifier les 20 principales anomalies
anomalies = np.argsort(anomaly_scores)[-50:]
print(f"Les 20 principales anomalies sont : {anomalies}")
