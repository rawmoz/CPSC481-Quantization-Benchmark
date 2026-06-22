import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("C:\\CPSC481\\results_gsm8k.csv")

# Model display order
model_order = ["deepseek-f16", "deepseek-q8", "deepseek-q6k", "deepseek-q4km", "deepseek-q2k"]
labels = ["F16", "Q8_0", "Q6_K", "Q4_K_M", "Q2_K"]
file_sizes = [16.07, 8.54, 6.60, 4.92, 3.18]

# Calculate stats
accuracy = []
avg_time = []
for model in model_order:
    m = df[df['model'] == model]
    accuracy.append(m['is_correct'].sum() / len(m) * 100)
    avg_time.append(m['time_seconds'].mean())

sns.set_theme(style="darkgrid")

# --- Graph 1: Accuracy Bar Chart ---
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracy, color=sns.color_palette("Blues_d", len(labels)))
plt.title("GSM8K Accuracy per Quantization Level", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for bar, val in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}%", ha='center', fontsize=12)
plt.tight_layout()
plt.savefig("C:\\CPSC481\\gsm8k_graph_accuracy.png", dpi=150)
plt.close()
print("Saved gsm8k_graph_accuracy.png")

# --- Graph 2: Response Time Bar Chart ---
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, avg_time, color=sns.color_palette("Oranges_d", len(labels)))
plt.title("GSM8K Average Response Time per Quantization Level", fontsize=16)
plt.xlabel("Model")
plt.ylabel("Avg Time per Question (seconds)")
for bar, val in zip(bars, avg_time):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val:.1f}s", ha='center', fontsize=12)
plt.tight_layout()
plt.savefig("C:\\CPSC481\\gsm8k_graph_time.png", dpi=150)
plt.close()
print("Saved gsm8k_graph_time.png")

# --- Graph 3: File Size vs Accuracy Scatter Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(file_sizes, accuracy, color='steelblue', s=200, zorder=5)
for i, label in enumerate(labels):
    plt.annotate(label, (file_sizes[i], accuracy[i]), textcoords="offset points", xytext=(10, 5), fontsize=12)
plt.title("GSM8K File Size vs Accuracy Tradeoff", fontsize=16)
plt.xlabel("File Size (GB)")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig("C:\\CPSC481\\gsm8k_graph_size_vs_accuracy.png", dpi=150)
plt.close()
print("Saved gsm8k_graph_size_vs_accuracy.png")

print("\nAll 3 GSM8K graphs saved to C:\\CPSC481")