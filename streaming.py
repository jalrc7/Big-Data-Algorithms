import numpy as np

#QUESTION 1
#part a
import matplotlib.pyplot as plt
from collections import Counter

def markov_chain_uniform_sample(n, s, steps):
    samples = []
    for _ in range(s):
        current_state = np.random.randint(1, n + 1)  
        for _ in range(steps):
            current_state = np.random.randint(1, n + 1)  
        samples.append(current_state)
    return samples

#part b
def D_far(n, s, epsilon):
    probabilities = np.ones(n) / n
    quarter_n = n // 4
    adjustment = (2 * epsilon) / (quarter_n * 2)
    for i in range(quarter_n):
        probabilities[i] += adjustment
        probabilities[-(i + 1)] -= adjustment
    probabilities /= probabilities.sum()  
    return np.random.choice(np.arange(1, n + 1), size=s, p=probabilities)


def count_collisions(samples):
    count = Counter(samples)
    return sum(c - 1 for c in count.values() if c > 1)


def find_s_and_t(n, epsilon, delta, steps):
    s = int(800 * np.sqrt(n / epsilon**2))  
    while True:
        U_collisions = [count_collisions(markov_chain_uniform_sample(n, s, steps)) for _ in range(int(3 / delta))]
        Dfar_collisions = [count_collisions(D_far(n, s, epsilon)) for _ in range(int(3 / delta))]

        U_collisions.sort()
        Dfar_collisions.sort()

        t_U = np.percentile(U_collisions, 100 * (1 - delta))
        t_Dfar = np.percentile(Dfar_collisions, 100 * delta)

        if t_U < t_Dfar:
            return s, (t_U + t_Dfar) / 2
        s = int(s * 1.1) 


n = 10000
epsilon = 0.2
delta = 0.1
steps = 1000

s, t = find_s_and_t(n, epsilon, delta, steps)
print(f"Sample size (s): {s}, Threshold (t): {t}")


#Question 3 
from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


def markov_chain_uniform_sample_categorical(categories, s):
    return np.random.choice(categories, size=s, replace=True)


def D_far_categorical(categories, s, epsilon):
    probabilities = np.ones(len(categories)) / len(categories)
    adjustments = epsilon / len(categories)
    probabilities[:len(categories) // 2] += adjustments
    probabilities[len(categories) // 2:] -= adjustments
    probabilities /= probabilities.sum()  
    return np.random.choice(categories, size=s, p=probabilities)


def count_collisions(samples):
    count = Counter(samples)
    return sum(c - 1 for c in count.values() if c > 1)


def find_s_and_t_categorical(categories, epsilon, delta, s_trials):
    s = int(800 * np.sqrt(len(categories) / epsilon**2))  
    while True:
        U_collisions = [count_collisions(markov_chain_uniform_sample_categorical(categories, s)) for _ in range(s_trials)]
        Dfar_collisions = [count_collisions(D_far_categorical(categories, s, epsilon)) for _ in range(s_trials)]

        U_collisions.sort()
        Dfar_collisions.sort()

        t_U = np.percentile(U_collisions, 100 * (1 - delta))
        t_Dfar = np.percentile(Dfar_collisions, 100 * delta)

        if t_U < t_Dfar:
            return s, (t_U + t_Dfar) / 2
        s = int(s * 1.1)  

categories = df['species'].unique()
epsilon = 0.1
delta = 0.1
s_trials = int(3 / delta)

s, t = find_s_and_t_categorical(categories, epsilon, delta, s_trials)
print(f"Sample size (s): {s}, Threshold (t): {t}")
actual_samples = df['species'].tolist()
actual_collisions = count_collisions(actual_samples)
print(f"Actual Collisions: {actual_collisions}")
print("Test Result:", actual_collisions < t)

def calculate_entropy(df):
    probabilities = df['species'].value_counts(normalize=True)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(len(probabilities))  
    print("Observed Entropy:", entropy)
    print("Maximum Possible Entropy:", max_entropy)
    return entropy / max_entropy  

normalized_entropy = calculate_entropy(df)
print("Normalized Entropy (1 indicates uniform distribution):", normalized_entropy)
