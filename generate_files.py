import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

def extract_center_coordinates(annotations):
    center_coordinates = []
    for annotation in annotations:
        bbox = annotation['bbox']
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        center_coordinates.append((x_center, y_center))
    return center_coordinates


def main():
    # Chemin vers le fichier JSON COCO
    json_file_path = '/home/edr/Desktop/Animal Herding Database/Goats-Geladas/kenyan-ungulates/ungulate-annotations/annotations-clean-name-pruned/annotations-clean-name-pruned.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    txt_file_path = f"animal_positions_global.csv"
    with open(txt_file_path, 'w') as csv_file:

        csv_file.write(f"image name; centroids\n")

        for image_data in data['images']:

            x_centroid, y_centroid = [], []
            image_id = image_data['id']
            image_file_name = image_data['file_name']

            # Récupération des annotations pour cette image
            image_annotations = [
                annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

            # Extraction des coordonnées du centre pour chaque annotation
            center_coordinates = extract_center_coordinates(image_annotations)
            x_coords, y_coords = zip(*center_coordinates)

            # Normalize the x and y coordinates
            normalized_x = [(x - min(x_coords)) / (max(x_coords) - min(x_coords)) for x in x_coords]
            normalized_y = [(y - min(y_coords)) / (max(y_coords) - min(y_coords)) for y in y_coords]
            
            # Perform DBSCAN clustering
            eps = 0.2  # The maximum distance between two samples to be considered as neighbors
            min_samples = 3  # The minimum number of samples in a neighborhood for a point to be considered as a core point
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(list(zip(normalized_x, normalized_y)))

            # Visualize the clusters
            unique_labels = set(labels)

            # Define a list of visible colors
            visible_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']

            # Assign a visible color to each label
            label_colors = {label: color for label, color in zip(unique_labels, visible_colors)}

            # Plot the clusters using visible colors
            for k in unique_labels:
                if k == -1:
                    # Outliers are plotted in black
                    col = 'black'
                else:
                    col = label_colors[k]

                class_member_mask = (labels == k)

                xy = np.array(list(zip(normalized_x, normalized_y)))[class_member_mask]
                plt.scatter(xy[:, 0], xy[:, 1], color=col, alpha=0.5)
            # Calculate the distribution of cluster sizes
            import scipy.stats as stats

            cluster_sizes = [np.sum(labels == k) for k in unique_labels]

            # Plot histogram of cluster sizes
            plt.hist(cluster_sizes, bins='auto')
            plt.xlabel('Cluster Size')
            plt.ylabel('Frequency')
            plt.title('Distribution of Cluster Sizes')
            plt.show()

            # Perform Shapiro-Wilk normality test
            _, p_value = stats.shapiro(cluster_sizes)
            if p_value > 0.05:
                print("Cluster sizes follow a Gaussian distribution.")
            else:
                print("Cluster sizes do not follow a Gaussian distribution.")

            # Print the mean, standard deviation, and minimum Mahalanobis distance to other clusters
            mean_cluster_size = np.mean(cluster_sizes)
            std_cluster_size = np.std(cluster_sizes)
            min_distance = np.min([np.min(np.ma.masked_where(k == j, np.linalg.norm(xy - np.mean(xy, axis=0), axis=1))) for k in unique_labels for j in unique_labels if k != j])

            print(f"Mean cluster size: {mean_cluster_size}")
            print(f"Standard deviation of cluster size: {std_cluster_size}")
            print(f"Minimum Mahalanobis distance to other clusters: {min_distance}")
            plt.xlabel('Normalized X')
            plt.ylabel('Normalized Y')
            plt.title('DBSCAN Clustering')
            #plt.show()


if __name__ == "__main__":
    main()
