import numpy as np
import matplotlib.pyplot as plt

def extract_features(piece_region):
    """Extrait les caractéristiques de l'image d'une pièce"""
    # Calculer la moyenne et l'écart-type des niveaux de gris
    mean = np.mean(piece_region)
    std = np.std(piece_region)
    
    # Calculer les quartiles pour avoir une meilleure distribution
    q1 = np.percentile(piece_region, 25)
    q2 = np.percentile(piece_region, 50)  # médiane
    q3 = np.percentile(piece_region, 75)

    
    iqr = q3 - q1  # Mesure de dispersion robuste
    
    # Ratio entre std et mean pour capturer le contraste relatif
    contrast_ratio = std / mean if mean > 0 else 0
    
    return np.array([mean, std, q1, q2, q3 ])


def classify_pieces(occupied_squares, debug=True):
    """
    Classifie les pièces en groupes en utilisant K-means
    occupied_squares: liste de tuples (piece_region, is_dark_square, position)
    """
    # Extraire les caractéristiques de chaque pièce
    features = []
    for piece_region, is_dark, pos in occupied_squares:
        feat = extract_features(piece_region)
        features.append(feat)
    
    features = np.array(features)
    
    # Normaliser les caractéristiques
    features_normalized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    # Add binary feature based on normalized std threshold
    std_threshold = np.array([1 if std > 28 else 0 for std in features[:, 1]])
    features_normalized = np.hstack((features_normalized, std_threshold.reshape(-1, 1)))

    # print(features_normalized)
    
    # Appliquer K-means avec 4 clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features_normalized)
    
    if debug:

        ########################################### 
        # * DISPLAY CLUSTERS RESULTS
        ###########################################

        # 1. Vue d'ensemble des clusters
        plt.figure(figsize=(20, 8))  # Figure plus large
        
        # 1.1 Afficher les échantillons de chaque cluster
        for cluster_id in range(4):
            cluster_samples = [i for i in range(len(clusters)) if clusters[i] == cluster_id]
            n_samples = len(cluster_samples)
            
            if n_samples > 0:
                # Calculer le nombre de colonnes nécessaires
                n_cols = n_samples + 1  # +1 pour le texte d'info
                
                # Texte d'information du cluster
                plt.subplot2grid((4, n_cols), (cluster_id, 0))
                plt.text(0.1, 0.5, f'Cluster {cluster_id}\n'
                        f'N: {n_samples}\n'
                        f'Mean: {np.mean([features[i][0] for i in cluster_samples]):.1f}\n'
                        f'Std: {np.mean([features[i][1] for i in cluster_samples]):.1f}',
                        fontsize=8)
                plt.axis('off')
                
                # Afficher tous les échantillons du cluster
                for idx, sample_idx in enumerate(cluster_samples):
                    plt.subplot2grid((4, n_cols), (cluster_id, idx + 1))
                    piece_region = occupied_squares[sample_idx][0]
                    plt.imshow(piece_region, cmap='gray')
                    pos = occupied_squares[sample_idx][2]
                    is_dark = occupied_squares[sample_idx][1]
                    std_val = features[sample_idx][1]  # Ajout de la valeur de std
                    plt.title(f'{pos}\n{"B" if is_dark else "W"}\nstd:{std_val:.1f}',  # Affichage du std
                            fontsize=6)
                    plt.xticks([])
                    plt.yticks([])
        
        plt.tight_layout(h_pad=0.1, w_pad=0.1, pad=0.5)  # Réduire les marges
        plt.show()

        ########################################### 
        # * DISPLAY FEATURES ANALYSIS
        ###########################################
        
        # 3. Visualisation détaillée des caractéristiques
        plt.figure(figsize=(15, 10))
        
        feature_names = ['Moyenne', 'Écart-type', 'q1', 'q2', 'q3' , 'Std_threshold']  # À adapter selon vos features
        n_features = len(feature_names)
        colors = ['red', 'blue', 'green', 'orange']
        
        # Calculer la disposition optimale pour les subplots
        n_plots = n_features + 1  # nombre total de plots (features + scatter)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols  # arrondi supérieur
        
        # 3.1 Box plots pour chaque feature
        for i, feature_name in enumerate(feature_names):
            plt.subplot(n_rows, n_cols, i + 1)
            data = [features_normalized[clusters == j, i] for j in range(4)]
            bp = plt.boxplot(data, labels=[f'Cluster {j}' for j in range(4)])
            
            # Colorer les box plots
            for j, box in enumerate(bp['boxes']):
                box.set(color=colors[j])
                bp['medians'][j].set(color='black', linewidth=2)
            
            plt.ylabel(feature_name)
            plt.title(f'Distribution de {feature_name}')
            plt.grid(True, alpha=0.3)
        
        # 3.2 Scatter plot des deux premières caractéristiques
        plt.subplot(n_rows, n_cols, n_features + 1)
        for cluster_id in range(4):
            cluster_mask = clusters == cluster_id
            plt.scatter(features[cluster_mask, 0], features[cluster_mask, 1],
                       label=f'Cluster {cluster_id}', color=colors[cluster_id])
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.legend()
        plt.title(f'{feature_names[0]} vs {feature_names[1]}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    
    # Analyse des clusters basée sur mean et std
    cluster_stats = []
    for c in range(4):
        cluster_mask = clusters == c
        mean_intensity = np.mean(features[cluster_mask, 0])  # Moyenne des means
        mean_std = np.mean(features[cluster_mask, 1])       # Moyenne des stds
        cluster_stats.append({
            'cluster': c,
            'mean': mean_intensity,
            'std': mean_std
        })
    
    # Trier les clusters par mean et std
    sorted_by_mean = sorted(cluster_stats, key=lambda x: x['mean'])
    sorted_by_std = sorted(cluster_stats, key=lambda x: x['std'])
    
    # Identifier les clusters avec mean élevé/faible et std élevé/faible
    high_mean_clusters = {stats['cluster'] for stats in sorted_by_mean[2:]}  # Les 2 plus hauts
    high_std_clusters = {stats['cluster'] for stats in sorted_by_std[2:]}    # Les 2 plus hauts
    
    if debug:
        print("\nAnalyse des clusters:")
        print("-" * 50)
        print("Classement par mean:")
        for i, stats in enumerate(sorted_by_mean):
            print(f"  {i+1}. Cluster {stats['cluster']}: {stats['mean']:.2f}")
        print("\nClassement par std:")
        for i, stats in enumerate(sorted_by_std):
            print(f"  {i+1}. Cluster {stats['cluster']}: {stats['std']:.2f}")
        print("\nGroupes identifiés:")
        print(f"Clusters avec mean élevé: {high_mean_clusters}")
        print(f"Clusters avec std élevé: {high_std_clusters}")
    
    # Détecter et retirer les outliers avant la classification finale
    def remove_outliers(cluster_id, features, clusters):
        cluster_mask = clusters == cluster_id
        cluster_stds = features[cluster_mask, 1]  # Récupérer les std du cluster
        
        # Calculer la moyenne des std du cluster
        mean_std = np.mean(cluster_stds)
        
        # Identifier les outliers (std très faible par rapport à la moyenne du cluster)
        outliers_mask = cluster_stds < (mean_std * 0.2)  # On peut ajuster ce seuil
        
        if debug and np.any(outliers_mask):
            print(f"\nOutliers détectés dans le cluster {cluster_id}:")
            print(f"Moyenne std du cluster: {mean_std:.2f}")
            outlier_positions = [occupied_squares[i][2] for i, is_outlier 
                               in enumerate(outliers_mask) if is_outlier]
            outlier_stds = cluster_stds[outliers_mask]
            for pos, std in zip(outlier_positions, outlier_stds):
                print(f"Position {pos}: std = {std:.2f} (ratio = {std/mean_std:.2%})")
        
        return outliers_mask
    
    # Détecter les outliers pour chaque cluster
    outliers = np.zeros(len(features), dtype=bool)
    for cluster_id in range(4):
        cluster_outliers = remove_outliers(cluster_id, features, clusters)
        cluster_mask = clusters == cluster_id
        outliers[cluster_mask] = cluster_outliers
    
    if debug:
        print("\nRésumé des outliers:")
        print(f"Nombre total d'outliers: {np.sum(outliers)}")
        outlier_positions = [occupied_squares[i][2] for i, is_outlier 
                           in enumerate(outliers) if is_outlier]
        print(f"Positions des outliers: {outlier_positions}")
    
    # Filtrer les résultats pour exclure les outliers
    results = {}
    for i, (_, _, pos) in enumerate(occupied_squares):
        if not outliers[i]:  # Ignorer les outliers
            cluster = clusters[i]
            is_high_mean = cluster in high_mean_clusters
            is_high_std = cluster in high_std_clusters
            
            if is_high_mean:
                piece_color = 'white' if is_high_std else 'black'
            else:
                piece_color = 'black' if is_high_std else 'white'
            
            results[pos] = piece_color
        else:
            # Pour les outliers, on ne met pas de couleur
            results[pos] = None
    
    return results


