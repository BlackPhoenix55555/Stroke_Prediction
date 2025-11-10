import csv
import numpy as np

def pca_from_cov(cov_matrix, data, variance_threshold=0.95):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues descending
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    k = np.argmax(cumulative_variance >= variance_threshold) + 1
    projection_matrix = eigenvectors[:, :k]
    
    data_centered = data - np.mean(data, axis=0)
    reduced_data = np.dot(data_centered, projection_matrix)
    
    return reduced_data, projection_matrix, k

def read_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                data.append([float(value) for value in row])
    return np.array(data)

def manual_covariance_matrix(data):
    n = data.shape[0]
    p = data.shape[1]
    means = np.mean(data, axis=0)
    cov_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            cov_matrix[i, j] = np.sum((data[:, i] - means[i]) * (data[:, j] - means[j])) / (n - 1)
    return cov_matrix

def save_to_csv(filename, data, target_column):
    # Append the target column to reduced data
    final_data = np.hstack((data, target_column.reshape(-1, 1)))
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in final_data:
            writer.writerow(row)
    print(f"Reduced data saved to '{filename}'")

def main(input_file, output_file="stroke_reduced.csv", variance_threshold=0.95):
    data = read_data(input_file)
    features = data[:, :-1]   # All columns except last
    target = data[:, -1]      # Last column (stroke)
    
    cov_matrix = manual_covariance_matrix(features)
    reduced_data, projection_matrix, k = pca_from_cov(cov_matrix, features, variance_threshold)
    
    save_to_csv(output_file, reduced_data, target)

if __name__ == "__main__":
    main("stroke_data_clean.csv", "stroke_reduced.csv")

