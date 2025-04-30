import matplotlib.pyplot as plt
import numpy as np
import os

# Create a folder to save plots
os.makedirs('plots', exist_ok=True)

# Updated model names
models = [
    "No Retrieval", "Single Retrieval", "IRCOT", "AdaptiveRAG (25 Epochs)",
    "TinyBERT", "MiniLM", "E5-Small-V2", "Deberta-v3-small", "Flan-T5-XL (LoRA)"
]

# Parameters (in millions)
params_million = [
    0.1, 770, 770, 770,
    14, 33, 110, 184, 3000
]

# Scores for each dataset
# Order: [No Retrieval, Single Retrieval, IRCOT, AdaptiveRAG, TinyBERT, MiniLM, E5-Small, Deberta-v3-small, Flan-T5-XL]

scores = {
    "Musique": {
        "F1": [7.8, 24.1, 32.6, 33.3, 31.7, 31.9, 32.6, 32.2, 32.0],
        "EM": [1.2, 14.6, 23.0, 24.4, 23.2, 23.2, 24.0, 23.2, 23.2]
    },
    "2wikimultihopqa": {
        "F1": [25.8, 37.3, 25.8, 51.8, 58.9, 56.31, 49.5, 51.66, 50.5],
        "EM": [22.6, 31.8, 22.6, 42.6, 49.6, 47.0, 40.6, 43.0, 41.6]
    },
    "HotpotQA": {
        "F1": [23.9, 47.2, 56.7, 55.99, 56.2, 61.8, 54.3, 51.95, 53.77],
        "EM": [16.0, 36.2, 43.4, 43.0, 44.6, 53.0, 42.4, 40.4, 41.8]
    },
    "NQ": {
        "F1": [21.3, 45.6, 44.7, 47.3, 47.2, 47.3, 47.3, 47.1, 47.2],
        "EM": [13.4, 33.0, 32.4, 37.8, 37.6, 37.8, 37.8, 37.4, 37.6]
    },
    "SQuAD": {
        "F1": [13.3, 40.5, 30.0, 38.6, 39.3, 39.1, 38.6, 35.6, 39.1],
        "EM": [5.8, 30.6, 39.6, 28.6, 28.0, 27.6, 27.0, 24.0, 27.8]
    },
    "Trivia": {
        "F1": [30.5, 62.8, 55.8, 60.6, 62.1, 61.8, 61.3, 58.7, 59.4],
        "EM": [25.8, 55.8, 62.5, 52.0, 53.2, 53.0, 52.6, 50.2, 50.4]
    }
}

# Plotting
for dataset, dataset_scores in scores.items():
    f1_scores = np.array(dataset_scores["F1"])
    em_scores = np.array(dataset_scores["EM"])
    
    # Plot F1
    plt.figure(figsize=(10,7))
    plt.scatter(params_million, f1_scores)
    for i, model in enumerate(models):
        plt.text(params_million[i]*1.05, f1_scores[i], model, fontsize=8)

    plt.xscale('log')
    plt.xlabel('Number of Parameters (Millions, Log Scale)')
    plt.ylabel('F1 Score (%)')
    plt.title(f'Parameters vs F1 Score ({dataset})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/parameters_vs_f1_score_{dataset}.png', dpi=300)
    plt.close()

    # Plot EM
    plt.figure(figsize=(10,7))
    plt.scatter(params_million, em_scores, color='orange')
    for i, model in enumerate(models):
        plt.text(params_million[i]*1.05, em_scores[i], model, fontsize=8)

    plt.xscale('log')
    plt.xlabel('Number of Parameters (Millions, Log Scale)')
    plt.ylabel('Exact Match (EM) Score (%)')
    plt.title(f'Parameters vs EM Score ({dataset})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/parameters_vs_em_score_{dataset}.png', dpi=300)
    plt.close()

print("All updated plots saved inside 'plots/' folder.")
