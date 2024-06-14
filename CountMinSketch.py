import numpy as np 
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from faker import Faker 
#Collaborators = NONE

#Problem 1: Implementation of CountMin Sketch
class CountMinSketch:
    def __init__(self, buckets, diffhash, p):
        self.buckets = buckets
        self.diffhash = diffhash
        self.p = p
        self.table = np.zeros((diffhash, buckets), dtype=np.int64)
        self.hash = [self.hash_generator(p, buckets) for _ in range(diffhash)]

    def update(self, item, count=1):
        for i, current in enumerate(self.hash):
            index = current(item)
            self.table[i, index] += count

    def query(self, item):
        estimates = [self.table[i, current(item)] for i, current in enumerate(self.hash)]
        return min(estimates)
    
    @staticmethod
    def hash_generator(p, buckets):
        a = random.randint(1, p - 1)
        b = random.randint(0, p - 1)
        return lambda x: ((a * hash(x) + b) % p) % buckets
    
    def estimated_freq(self, items):
        frequencies = {}
        for item in items:
            frequencies[item] = self.query(item)
        return frequencies
    
#Problem 2, testing on a artifical dataset
p = 100003
buckets =  53
diffhash = 5
cms = CountMinSketch(p, buckets, diffhash)
dataset = ['apple'] * 100 + ['banana'] * 300 + ['cherry'] * 200 + ['date'] * 120 + ['elderberry'] * 250

for item in dataset:
    cms.update(item)

items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
frequencies = cms.estimated_freq(items)
for item, freq in frequencies.items():
    print(f"{item}: Estimated Freq = {freq}")
    #Output
    #apple: Estimated Freq = 100
    #banana: Estimated Freq = 500
    #cherry: Estimated Freq = 500
    #date: Estimated Freq = 370
    #elderberry: Estimated Freq = 370

#Problem 3
class UniversalHashing:
    def __init__(self,p, length):
        self.p = p 
        self.length = length
        self.a = [random.randint(0, p-1) for _ in range(length)]
        self.b = random.randint(0, p - 1)
    
    def _pad_or_chunk_string(self, string):
        array_string = [ord(char) for char in string]

        if len(array_string) ==  self.length:
            return [array_string]
        chunks = []
        if len(array_string) < self.length:
        # If the string is shorter or equal to the expected length, pad it with zeros
            padding = self.length - len(array_string)
            padded_chunk = array_string + [0] * padding
            chunks.append(padded_chunk)
        else:
        # If the string is longer, break it into chunks of length self.length
            for i in range(0, len(array_string), self.length):
            # Take a slice of self.length, starting from index i
                chunk =array_string[i:i + self.length]
                if len(chunk) < self.length: 
                    chunk.extend([0] * (self.length - len(chunk)))
                chunks.append(chunk)
        return chunks


    def hash(self, string):
        chunks = self._pad_or_chunk_string(string)
        hash_value = 0
        for chunk in chunks:
            chunk_hash = self.b
            for i, digit in enumerate(chunk):
                chunk_hash += self.a[i] * digit
            hash_value = (hash_value * self.p + chunk_hash) % self.p

        return hash_value

    def rehash(self):

        self.a = [random.randint(0, self.p - 1) for _ in range(self.length)]
        self.b = random.randint(0, self.p - 1)

#Problem 4 - testing hash functions
df = pd.read_csv("/Users/jalrc/DS 563/data-2.csv", usecols=['title'])
p = 100003
length = max(df['title'].str.len()) 
hashing = UniversalHashing(p, length)
hash_counts = defaultdict(int)
unique_words = defaultdict(set)

for title in df['title']:
    words = title.lower().split() 
    for word in words:
        hash_value = hashing.hash(word)  # Hash each word
        hash_counts[hash_value] += 1
        unique_words[hash_value].add(word)  # Track unique words for each hash value

# Calculating overestimation for each hash value
overestimation = {hv: hash_counts[hv] - len(unique_words[hv]) for hv in hash_counts}

# Plotting 
overestimation_values = list(overestimation.values())
plt.figure(figsize=(12, 6))
plt.hist(overestimation_values, bins=range(min(overestimation_values), max(overestimation_values) + 1), color='skyblue', edgecolor='black')
plt.xlabel('Overestimation Amount')
plt.ylabel('Number of Hash Values')
plt.title('Histogram of Hash Value Overestimation for TED Talk Title Words')
plt.tight_layout()
plt.show()

#Question 5- Experiements on CountMinSketch
#Resources Used:
# https://www.datacamp.com/tutorial/creating-synthetic-data-with-python-faker-tutorial

#Part a
p = 100003
def generate_faker_dataset(num_samples):
    fake = Faker()
    return [fake.word() for _ in range(num_samples)]  # Using 'word' for more variability in the dataset

# Adjusting the testing function to match corrected initialization
def testing(dataset, N, K, p=100003):
    # Corrected order of parameters when initializing CountMinSketch
    cms = CountMinSketch(K, N, p)
    for item in dataset:
        cms.update(item)
    part = np.random.choice(dataset, size=100, replace=False)
    actual_freq = {item: dataset.count(item) for item in part}
    esti_freq = {item: cms.query(item) for item in part}
    errors = [abs(actual_freq[item] - esti_freq[item]) for item in part]
    avg_error = np.mean(errors)
    
    print(f"Configuration (N={N}, K={K}): Average Error = {avg_error:.2f}")

samples = 10000
dataset = generate_faker_dataset(samples)
configurations = [(100, 100), (200, 50), (50, 200)]
for N, K in configurations:
    testing(dataset, N, K, p)

#Output 
#Configuration (N=100, K=100): Average Error = 25.35, showing decent accuracy 
#Configuration (N=200, K=50): Average Error = 85.09, worst more hash functions with fewwer buckets cause more mistakes 
#Configuration (N=50, K=200): Average Error = 6.27, best accuracy suggets that fewer hash functions but more buckets support better accuracy
    
#PART B
# Our analysis predicts that the more we can spread our items across buckets, the more we can reduce errors such as collisions.
# The observed behavior does match our predications as seen in part A with Configuration (N=200, K=50) and Configuration (N=50, K=200):

#Question 6 Experiment
def run_countminsketch_experiment():

    p = 100003
    buckets = 53
    diffhash = 5
    cms = CountMinSketch(buckets, diffhash, p)
    dataset = (['apple'] * 100) + (['banana'] * 300) + (['cherry'] * 200) + (['date'] * 120) + (['elderberry'] * 250)

    for item in dataset:
        cms.update(item)

    items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    actual_freq = {'apple': 100, 'banana': 300, 'cherry': 200, 'date': 120, 'elderberry': 250}
    estimated_freq = {item: cms.query(item) for item in items}

    S = sum(actual_freq.values())
    threshold = 2 * S / buckets
    overestimations = [estimated_freq[item] - actual_freq[item] for item in items]
    over_threshold_count = len([over for over in overestimations if over >= threshold])
    prob_over_threshold = over_threshold_count / len(items) 

    print(f"Probability of overestimation >= 2S/K among selected items: {prob_over_threshold:.4f}")

    plt.bar(items, overestimations, color='skyblue')
    plt.xlabel('Item')
    plt.ylabel('Overestimation Amount')
    plt.title('Overestimation of Item Frequencies in CountMinSketch')
    plt.show()

if __name__ == "__main__":
    run_countminsketch_experiment()
#For some reason I get 0 as the output which we know is not true
