import KitNET.KitNET as kit
import numpy as np
import pandas as pd
import time

data_path = "/home/milan/Desktop/Elfak_praksa/Kitsune-py/archive/ARP_MitM_dataset.csv"

X = pd.read_csv(data_path, header=None)
label_path = "/home/milan/Desktop/Elfak_praksa/Kitsune-py/archive/ARP_MitM_labels.csv"
Y = pd.read_csv(label_path)
print(np.unique(Y['x'].values))

filter_data = X.copy()
filter_data['y'] = Y['x']
filter_data.head(10)

benign = X[filter_data['y']==0].iloc[:50000]


malicious = X[filter_data['y']==1][:50000]

all_data = pd.concat([benign, malicious])

maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 1000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 10000 #the number of instances used to train the anomaly detector (ensemble itself)

K = kit.KitNET(all_data.shape[1],maxAE,FMgrace,ADgrace)
RMSEs = np.zeros(all_data.shape[0]) # a place to save the scores


print("Running KitNET:")
start = time.time()

timestamps = []
for i in range(all_data.shape[0]):
    timestamps.append(time.time()-start)
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(all_data.iloc[i,]) #will train during the grace periods, then execute on all the rest.
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm
benignSample = np.log(RMSEs[FMgrace+ADgrace+1:71000])
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))
timestamps = np.array(timestamps)
# plot the RMSE anomaly scores
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,5))
fig = plt.scatter(timestamps[FMgrace+ADgrace+1:],RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from KitNET's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
plt.annotate('Mirai C&C channel opened [Telnet]', xy=(timestamps[71662],RMSEs[71662]), xytext=(timestamps[58000],1),arrowprops=dict(facecolor='black', shrink=0.05),)
plt.annotate('Mirai Bot Activated\nMirai scans network for vulnerable devices', xy=(timestamps[72662],1), xytext=(timestamps[55000],5),arrowprops=dict(facecolor='black', shrink=0.05),)
figbar=plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()