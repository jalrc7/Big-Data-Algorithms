
import numpy as np 
import pandas as pd 


from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
def jl_implentation(x, dimension_target, transform = "gaussian"):
    k =  x.shape[0]
    alpha = 1 / np.sqrt(dimension_target)
    if transform  == "gaussian":
        M = np.random.normal(loc = 0.0, scale = 1.0, size =(dimension_target, k))
    elif transform   == "radamacher":
        M = np.random.choice([-1, 1], size=(dimension_target, k))
    elif transform  == "probability":
        distribution = [-1, 0, 1]
        probs = [1/6, 2/3, 1/6]
        M = np.random.choice(distribution, size =(dimension_target, k),  p = probs)
    elif transform  == "one-third":
        M = np.zeros(shape = (dimension_target, k))
        sample_size = min(int(np.ceil(k / 3)), dimension_target, k)
        for column in range(k):
            chosen = np.random.choice(range(dimension_target), size = sample_size, replace = False)
            M[chosen, column ] = np.random.choice([-1, 1], size=sample_size)
    JL_transformation = alpha * np.dot(M, x)
    return JL_transformation

#problem 2 Proposed measures

#function to handle three proposed measures, avg_divergence and variance divergence and seeing the five nearest neighbors


def divergence_measures(x, dimension_target, transform,  n_neighbors=5 ):
    n = len(x)
    divergence =[]
    neighborhood_preservation_scores = []

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(x.T)
    _, before = nbrs.kneighbors(x.T)
    X_transformed = np.array([jl_implentation(x[:, i], dimension_target, transform) for i in range(n)]).T
    nbrs_transformed = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_transformed.T)
    _, after = nbrs_transformed.kneighbors(X_transformed.T)
    for i in range(n):
        #finding differences between original data and transformed data 
        overlapping = len(set(before[i]) & set(after[i])) - 1
        neighborhood_preservation_scores.append(overlapping / n_neighbors)
        for j in range(i + 1, n ):
            u = x[:,i]
            v = x[:,j]
            u_transform = jl_implentation(u, dimension_target , transform )
            v_transform = jl_implentation(v, dimension_target, transform)
            org_diff = np.linalg.norm(u - v )
            new_diff = np.linalg.norm(u_transform  - v_transform)
            if org_diff != 0: #in order to make sure they aren't same point
                ratio = new_diff / org_diff 
                divergence.append(ratio)
            

    if divergence:
        avg_divergence = np.mean(divergence)
        variance_divergence = np.var(divergence)
    if neighborhood_preservation_scores:
        avg_neighborhood_preservation = np.mean(neighborhood_preservation_scores)
    return avg_divergence, variance_divergence, avg_neighborhood_preservation

#Problem 3 comparisons

#Dataset 1- Genomic data + Preprocessing
print("Script started")
counts=pd.read_table('/Users/jalrc/Downloads/GSE157103_genes.tsv', sep='\t') 
counts = counts.rename(columns={'#symbol': 'sampleID'}) 
counts = counts.set_index('sampleID').T
expression_counts = (counts > 0).sum(axis=0)
nonzero_filter = counts.loc[:,expression_counts >= (len(counts) * 0.1)] 
columns = nonzero_filter.shape
print("genome dataset has {columns}")


dimension_targets = [40, 80, 100]
print("dimension_targets")
transforms = ['gaussian', 'radamacher', 'probability', 'one-third']
results = {t: {'avg_divergence': [], 'variance_divergence': [], 'avg_neighborhood_pres': []} for t in transforms}

for d in dimension_targets:
    for transform in transforms:
        transform_data = jl_implentation(nonzero_filter, d, transform)
        avg_div, var_div, avg_neighborhood_preservation  = divergence_measures(transform_data, d, transform)
        results[transform]['avg_divergence'].append(avg_div)
        results[transform]['variance_divergence'].append(var_div)
        results[transform]['avg_neighborhood_pres'].append(avg_neighborhood_preservation)
        print("Completed transform={transform}")
    print(f"Completed d={d}")
#showing transformations
for metric in ['avg_divergence']:
    plt.figure(figsize=(10, 6))
    for transform in transforms:
        plt.plot(dimension_targets, results[transform][metric], '-o', label=transform)
    plt.title(f'JL Transformation Analysis - {metric.replace("_", " ").title()}')
    plt.ylabel(metric)
    plt.xlabel('Target Dimension (d)')
    plt.legend()
    plt.show()
for metric in ['variance_divergence']:
    plt.figure(figsize=(10, 6))
    for transform in transforms:
        plt.plot(dimension_targets, results[transform][metric], '-o', label=transform)
    plt.title(f'JL Transformation Analysis - {metric.replace("_", " ").title()}')
    plt.ylabel(metric)
    plt.xlabel('Target Dimension (d)')
    plt.legend()
    plt.show()
for metric in [ 'avg_neighborhood_pres']:
    plt.figure(figsize=(10, 6))
    for transform in transforms:
        plt.plot(dimension_targets, results[transform][metric], '-o', label=transform)
    plt.title(f'JL Transformation Analysis - {metric.replace("_", " ").title()}')
    plt.ylabel(metric)
    plt.xlabel('Target Dimension (d)')
    plt.legend()
    plt.show()

#Dataset 2 - MNIST image processing 
#for simplicity only using training data

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16) #only need first 16 bytes for images + dimensions
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28 * 28) 

images_data= '/Users/jalrc/Downloads/train-images.idx3-ubyte'
train_images = load_mnist_images(images_data)


dimension_targets = [40,80, 100]
print("dimension_targets")
transforms = ['gaussian', 'radamacher', 'probability', 'one-third']
results = {t: {'avg_divergence': [], 'variance_divergence': [], 'avg_neighborhood_pres': []} for t in transforms}

for d in dimension_targets:
    for transform in transforms:
        transform_data = jl_implentation(train_images, d, transform)
        avg_div, var_div, avg_neighborhood_preservation  = divergence_measures(transform_data, d, transform)
        results[transform]['avg_divergence'].append(avg_div)
        results[transform]['variance_divergence'].append(var_div)
        results[transform]['avg_neighborhood_pres'].append(avg_neighborhood_preservation)
        print("Completed transform={transform}")
    print(f"Completed d={d}")
#showing transformations
for metric in ['avg_divergence']:
    plt.figure(figsize=(10, 6))
    for transform in transforms:
        plt.plot(dimension_targets, results[transform][metric], '-o', label=transform)
    plt.title(f'JL Transformation Analysis - {metric.replace("_", " ").title()}')
    plt.ylabel(metric)
    plt.xlabel('Target Dimension (d)')
    plt.legend()
    plt.show()
for metric in ['variance_divergence']:
    plt.figure(figsize=(10, 6))
    for transform in transforms:
        plt.plot(dimension_targets, results[transform][metric], '-o', label=transform)
    plt.title(f'JL Transformation Analysis - {metric.replace("_", " ").title()}')
    plt.ylabel(metric)
    plt.xlabel('Target Dimension (d)')
    plt.legend()
    plt.show()
for metric in [ 'avg_neighborhood_pres']:
    plt.figure(figsize=(10, 6))
    for transform in transforms:
        plt.plot(dimension_targets, results[transform][metric], '-o', label=transform)
    plt.title(f'JL Transformation Analysis - {metric.replace("_", " ").title()}')
    plt.ylabel(metric)
    plt.xlabel('Target Dimension (d)')
    plt.legend()
    plt.show()
