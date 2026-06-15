from Kitsune import Kitsune
import numpy as np
import time
from scipy.stats import norm
from matplotlib import pyplot as plt
import csv
# ==============================================
# 0. Settings
# ==============================================
path = "archive/SYN_DoS_pcap.pcap"
packet_limit = 100000   
maxAE = 10
FMgrace = 5000
ADgrace = 50000

# Final threshold percentile (adjust based on sweep results)
FINAL_PERCENTILE = 95   # 95 = high recall, still low FPR

# ==============================================
# 1. Run Kitsune
# ==============================================
print("Building Kitsune...")
K = Kitsune(path, packet_limit, maxAE, FMgrace, ADgrace)

print("Processing packets...")
RMSEs = []
i = 0
start = time.time()
features_output_path = "first_1000_features.csv"
with open(features_output_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    headers = K.FE.nstat.getNetStatHeaders()
    csv_writer.writerow(headers)

    while True:
        i += 1
        # Adjusted print frequency so you can actually see the progress for just 1000 packets
        if i % 100 == 0:
            print(f"Processing packet {i}/1000...")
        
        # 1. Extract the feature vector
        x = K.FE.get_next_vector()
        
        # 2. Check if EOF (no vector returned)
        if len(x) == 0:
            break
        if i <= 1000:
            csv_writer.writerow(x)
            
        # 4. Process the vector through KitNET
        rmse = K.AnomDetector.process(x)
        RMSEs.append(rmse)

        # 5. HARD STOP after 1000 lines
        

stop = time.time()
print(f"Complete. Time elapsed: {stop - start:.2f} s")

# ==============================================
# 2. Ground Truth & Benign Baseline
# ==============================================
total_packets = len(RMSEs)
train_end = FMgrace + ADgrace            # 55,000
benign_end = 70000                        # Mirai attack starts here
exec_start = train_end + 1

# Pure benign RMSEs for threshold calibration
benign_exec_RMSEs = RMSEs[exec_start:benign_end]

# Execution phase (all packets after training)
RMSEs_exec = RMSEs[exec_start:]
labels_exec = np.zeros(len(RMSEs_exec), dtype=bool)
labels_exec[benign_end - exec_start:] = True   # True = attack

# ==============================================
# 3. Threshold Sweep (uncomment to see all options)
# ==============================================
print("\n" + "=" * 60)
print("            THRESHOLD SWEEP")
print("=" * 60)
sweep_results = []
for p in [99.9, 99.5, 99, 98, 95, 90, 80]:
    th = np.percentile(benign_exec_RMSEs, p)
    preds = np.array(RMSEs_exec) > th
    tp = np.sum(preds & labels_exec)
    fp = np.sum(preds & ~labels_exec)
    tn = np.sum(~preds & ~labels_exec)
    fn = np.sum(~preds & labels_exec)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    sweep_results.append((p, th, tp, fp, tn, fn, rec, fpr, prec))
    print(f"Perc {p:5.1f} | Thresh {th:.6f} | Recall {rec:.4%} | FPR {fpr:.4%} | Precision {prec:.4%}")
print("=" * 60)

# ==============================================
# 4. Final Evaluation (using chosen percentile)
# ==============================================
threshold = np.percentile(benign_exec_RMSEs, FINAL_PERCENTILE)
predictions = np.array(RMSEs_exec) > threshold

TP = np.sum(predictions & labels_exec)
FP = np.sum(predictions & ~labels_exec)
TN = np.sum(~predictions & ~labels_exec)
FN = np.sum(~predictions & labels_exec)

recall = TP / (TP + FN) if (TP + FN) else 0.0
fpr = FP / (FP + TN) if (FP + TN) else 0.0
precision = TP / (TP + FP) if (TP + FP) else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

# ==============================================
# 5. Write Results to File
# ==============================================
output_file = "detection_metrics.txt"
with open(output_file, 'w') as f:
    f.write("Kitsune Detection Metrics (Mirai pcap)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total packets processed: {total_packets}\n")
    f.write(f"Training packets (FM+AD): {train_end}\n")
    f.write(f"Execution packets: {len(RMSEs_exec)}\n")
    f.write(f"Threshold ({FINAL_PERCENTILE}th percentile): {threshold:.6f}\n\n")
    
    f.write("Final Metrics:\n")
    f.write(f"True Positives  (TP): {TP}\n")
    f.write(f"False Positives (FP): {FP}\n")
    f.write(f"True Negatives  (TN): {TN}\n")
    f.write(f"False Negatives (FN): {FN}\n")
    f.write("-" * 40 + "\n")
    f.write(f"Recall (TPR):         {recall:.4%}\n")
    f.write(f"False Positive Rate:  {fpr:.4%}\n")
    f.write(f"Precision:            {precision:.4%}\n")
    f.write(f"F1-Score:             {f1:.4%}\n")
    f.write("=" * 50 + "\n\n")

    # Optional: include the sweep results
    f.write("Threshold Sweep (for reference):\n")
    f.write(f"{'Perc':>6} {'Thresh':<10} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Recall':<10} {'FPR':<10} {'Precision':<10}\n")
    for (p, th, tp, fp, tn, fn, rec, fpr, prec) in sweep_results:
        f.write(f"{p:5.1f}  {th:<10.6f} {tp:<8} {fp:<8} {tn:<8} {fn:<8} {rec:<10.4%} {fpr:<10.4%} {prec:<10.4%}\n")

print(f"\nMetrics saved to {output_file}")

# ==============================================
# 6. Plot (execution phase)
# ==============================================
benign_log = np.log(benign_exec_RMSEs)
logProbs = norm.logsf(np.log(RMSEs), np.mean(benign_log), np.std(benign_log))

plt.figure(figsize=(10, 5))
plt.scatter(
    range(train_end + 1, total_packets),
    RMSEs[train_end + 1:],
    s=0.1,
    c=logProbs[train_end + 1:],
    cmap='RdYlGn'
)
plt.yscale("log")
plt.title("Anomaly Scores (Execution Phase)")
plt.ylabel("RMSE (log scale)")
plt.xlabel("Packet index")
cbar = plt.colorbar()
cbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()