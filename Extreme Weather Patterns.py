import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from datasketch import MinHash, MinHashLSH

def misra_gries(stream, k):
    counters = {}
    for item in stream:
        if item in counters:
            counters[item] += 1
        elif len(counters) < k:
            counters[item] = 1
        else:
            min_count = min(counters.values())
            counters = {x: counters[x] - min_count for x in list(counters) if counters[x] > min_count}
    return counters

def evaluate_best_k(df, column):
    k_values = [5, 10, 15, 20, 25, 30]
    best_k = k_values[0]
    best_score = float('inf')
    for k in k_values:
        scores = []
        for _ in range(100):
            sample = df[column].dropna().sample(frac=1, replace=True)
            estimated_counts = misra_gries(sample, k)
            score = np.var(list(estimated_counts.values()))
            scores.append(score)
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_k = k
    return best_k

def fill_with_misra_gries(df, column, k):
    stream = df[column].dropna()
    freq_items = misra_gries(stream, k)
    if freq_items:
        most_common = max(freq_items, key=freq_items.get)
        df[column] = df[column].fillna(most_common)
    return df

def fill_missing_values(df):
    columns_to_process = ['PRCP', 'SNOW', 'TMAX', 'TMIN', 'WT08', 'WT01', 'WT16']
    print("Null counts before filling:")
    print_null_counts(df, columns_to_process)
    for column in columns_to_process:
        if column in df.columns:
            best_k = evaluate_best_k(df, column)
            print(f"Best k for {column}: {best_k}")
            df = fill_with_misra_gries(df, column, best_k)
    print("Null counts after filling:")
    print_null_counts(df, columns_to_process)
    return df

def print_null_counts(df, columns):
    null_counts = df[columns].isna().sum()
    print(null_counts)

def load_and_preprocess(directory):
    all_weather_data = []
    required_columns = ['PRCP', 'SNOW', 'TMAX', 'TMIN', 'WT08', 'WT01', 'WT16', 'STATION', 'DATE', 'MONTH', 'YEAR']
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = [col.strip().upper() for col in df.columns]
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df[df['DATE'].dt.year >= 1958]
            df['MONTH'] = df['DATE'].dt.month.astype(int)
            df['YEAR'] = df['DATE'].dt.year
            df = df[['STATION', 'DATE', 'MONTH', 'YEAR'] + [col for col in required_columns if col in df.columns]]
            if not df.empty:
                all_weather_data.append(df)
    combined_df = pd.concat(all_weather_data, ignore_index=True)
    return combined_df

class QuantileSummary:
    def __init__(self, eps):
        self.eps = eps
        self.tuples = []
        self.N = 0

    def insert(self, value):
        rank_min = 0
        self.N += 1
        new_tuple = (value, rank_min, self.eps * self.N)
        if not self.tuples:
            self.tuples.append(new_tuple)
            return
        for i, tup in enumerate(self.tuples):
            if value < tup[0]:
                self.tuples.insert(i, new_tuple)
                break
        else:
            self.tuples.append(new_tuple)
        self.compress()

    def compress(self):
        if len(self.tuples) < 2:
            return
        compressed = [self.tuples[0]]
        for i in range(1, len(self.tuples)):
            (v, r_min, delta) = self.tuples[i]
            (v_prev, r_min_prev, delta_prev) = compressed[-1]
            if r_min - r_min_prev + delta + delta_prev + 1 <= self.eps * self.N:
                compressed[-1] = (v_prev, r_min_prev, delta + delta_prev + 1)
            else:
                compressed.append((v, r_min, delta))
        self.tuples = compressed

    def query(self, quantile):
        target_rank = quantile * self.N
        rank_sum = 0
        for value, r_min, delta in self.tuples:
            rank_sum += r_min
            if rank_sum + delta >= target_rank:
                return value
        return None

def analyze_extreme_weather(data_directory):
    quantile_summary = QuantileSummary(eps=0.01)  # 1% error in quantile
    required_columns = ['PRCP', 'SNOW', 'TMAX', 'TMIN', 'WT08', 'WT01', 'WT16', 'STATION', 'DATE', 'MONTH', 'YEAR']
    all_data = load_and_preprocess(data_directory)
    for index, row in all_data.iterrows():
        if not pd.isna(row['PRCP']):
            quantile_summary.insert(row['PRCP'])
    extreme_value = quantile_summary.query(0.90)
    print(f"The 90th percentile of precipitation is approximately {extreme_value}mm.")
    return extreme_value

# Main Execution Flow
directory_path = '/content/drive/MyDrive/Newdata'
weather_data = load_and_preprocess(directory_path)
weather_data = fill_missing_values(weather_data)
extreme_precipitation = analyze_extreme_weather(directory_path)
