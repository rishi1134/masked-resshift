import re
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):

    if window_size <= 1:
        return data  

    smoothed_data = []
    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)c:\Users\yash9\Documents\GitHub\ResShift\plot.py
        end_index = min(len(data), i + window_size // 2 + 1)
        smoothed_data.append(np.mean(data[start_index:end_index]))

    return smoothed_data


with open('nohup_theirs_.txt', 'r') as file:
    content = file.read()

t_train_lines = [line for line in content.splitlines() if line.startswith('Train:')]

t_train_steps = []
t_train_t1_mse = []
t_train_t1_lpips = []
t_train_t3_mse = []
t_train_t3_lpips = []
t_train_t4_mse = []
t_train_t4_lpips = []

for line in t_train_lines:
    match = re.search(
        r'Train: (\d+)/\d+, MSE/LPIPS: t\(1\):([\d.e+-]+)/([\d.e+-]+), t\(3\):([\d.e+-]+)/([\d.e+-]+), t\(4\):([\d.e+-]+)/([\d.e+-]+)',
        line)
    if match:
        t_train_steps.append(int(match.group(1)))
        t_train_t1_mse.append(float(match.group(2)))
        t_train_t1_lpips.append(float(match.group(3)))
        t_train_t3_mse.append(float(match.group(4)))
        t_train_t3_lpips.append(float(match.group(5)))
        t_train_t4_mse.append(float(match.group(6)))
        t_train_t4_lpips.append(float(match.group(7)))

t_val_psnr = []
t_val_lpips = []
t_val_lines = [line for line in content.splitlines() if line.startswith('Validation Metric:')]
for line in t_val_lines:
    match = re.search(r'PSNR=([\d.]+), LPIPS=([\d.]+)', line)
    if match:
        t_val_psnr.append(float(match.group(1)))
        t_val_lpips.append(float(match.group(2).split('...')[0]))

t_val_steps = list(range(len(t_val_psnr)))


with open('nohup_ours_f.txt', 'r') as file:
    content = file.read()

train_lines = [line for line in content.splitlines() if line.startswith('Train:')]


train_steps = []
train_t1_mse = []
train_t1_lpips = []
train_t3_mse = []
train_t3_lpips = []
train_t4_mse = []
train_t4_lpips = []


for line in train_lines:
    match = re.search(
        r'Train: (\d+)/\d+, MSE/LPIPS: t\(1\):([\d.e+-]+)/([\d.e+-]+), t\(3\):([\d.e+-]+)/([\d.e+-]+), t\(4\):([\d.e+-]+)/([\d.e+-]+)',
        line)
    if match:
        train_steps.append(int(match.group(1)))
        train_t1_mse.append(float(match.group(2)))
        train_t1_lpips.append(float(match.group(3)))
        train_t3_mse.append(float(match.group(4)))
        train_t3_lpips.append(float(match.group(5)))
        train_t4_mse.append(float(match.group(6)))
        train_t4_lpips.append(float(match.group(7)))

val_psnr = []
val_lpips = []
val_lines = [line for line in content.splitlines() if line.startswith('Validation Metric:')]
for line in val_lines:
    match = re.search(r'PSNR=([\d.]+), LPIPS=([\d.]+)', line)
    if match:
        val_psnr.append(float(match.group(1)))
        val_lpips.append(float(match.group(2).split('...')[0]))

val_steps = list(range(len(val_psnr)))


with open('nohup_ours_4.txt', 'r') as file:
    content = file.read()


o_train_lines = [line for line in content.splitlines() if line.startswith('Train:')]


o_train_steps = []
o_train_t1_mse = []
o_train_t1_lpips = []
o_train_t3_mse = []
o_train_t3_lpips = []
o_train_t4_mse = []
o_train_t4_lpips = []


for line in o_train_lines:
    match = re.search(
        r'Train: (\d+)/\d+, MSE/LPIPS: t\(1\):([\d.e+-]+)/([\d.e+-]+), t\(3\):([\d.e+-]+)/([\d.e+-]+), t\(4\):([\d.e+-]+)/([\d.e+-]+)',
        line)
    if match:
        o_train_steps.append(int(match.group(1)))
        o_train_t1_mse.append(float(match.group(2)))
        o_train_t1_lpips.append(float(match.group(3)))
        o_train_t3_mse.append(float(match.group(4)))
        o_train_t3_lpips.append(float(match.group(5)))
        o_train_t4_mse.append(float(match.group(6)))
        o_train_t4_lpips.append(float(match.group(7)))


o_val_psnr = []
o_val_lpips = []
o_val_lines = [line for line in content.splitlines() if line.startswith('Validation Metric:')]
for line in o_val_lines:
    match = re.search(r'PSNR=([\d.]+), LPIPS=([\d.]+)', line)
    if match:
        o_val_psnr.append(float(match.group(1)))
        o_val_lpips.append(float(match.group(2).split('...')[0]))

o_val_steps = list(range(len(o_val_psnr)))


window_size_1 = 20
window_size_2 = 20

# Apply moving average to training data
smoothed_train_t1_mse = moving_average(train_t1_mse, window_size_1)
smoothed_train_t3_mse = moving_average(train_t3_mse, window_size_1)
smoothed_train_t4_mse = moving_average(train_t4_mse, window_size_1)
smoothed_t_train_t1_mse = moving_average(t_train_t1_mse, window_size_1)
smoothed_t_train_t3_mse = moving_average(t_train_t3_mse, window_size_1)
smoothed_t_train_t4_mse = moving_average(t_train_t4_mse, window_size_1)
smoothed_o_train_t1_mse = moving_average(o_train_t1_mse, window_size_1)
smoothed_o_train_t3_mse = moving_average(o_train_t3_mse, window_size_1)
smoothed_o_train_t4_mse = moving_average(o_train_t4_mse, window_size_1)

smoothed_train_t1_lpips = moving_average(train_t1_lpips, window_size_2)
smoothed_train_t3_lpips = moving_average(train_t3_lpips, window_size_2)
smoothed_train_t4_lpips = moving_average(train_t4_lpips, window_size_2)
smoothed_t_train_t1_lpips = moving_average(t_train_t1_lpips, window_size_2)
smoothed_t_train_t3_lpips = moving_average(t_train_t3_lpips, window_size_2)
smoothed_t_train_t4_lpips = moving_average(t_train_t4_lpips, window_size_2)
smoothed_o_train_t1_lpips = moving_average(o_train_t1_lpips, window_size_2)
smoothed_o_train_t3_lpips = moving_average(o_train_t3_lpips, window_size_2)
smoothed_o_train_t4_lpips = moving_average(o_train_t4_lpips, window_size_2)

# Apply moving average to validation data
smoothed_val_psnr = moving_average(val_psnr, window_size_1)
smoothed_val_lpips = moving_average(val_lpips, window_size_2)
smoothed_t_val_psnr = moving_average(t_val_psnr, window_size_1)
smoothed_t_val_lpips = moving_average(t_val_lpips, window_size_2)
smoothed_o_val_psnr = moving_average(o_val_psnr, window_size_1)
smoothed_o_val_lpips = moving_average(o_val_lpips, window_size_2)


# Plot smoothed training MSE
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)

plt.plot(train_steps, smoothed_train_t4_mse, label='App1 t4 MSE (Smoothed)')
plt.plot(o_train_steps, smoothed_o_train_t4_mse, label='App2 t4 MSE (Smoothed)')

plt.plot(t_train_steps, smoothed_t_train_t4_mse, label='Original t4 MSE (Smoothed)')
plt.xlabel('Training Step')
plt.ylabel('MSE')
plt.title(f'Training MSE')
plt.legend()

# Plot smoothed training LPIPS
plt.subplot(1, 2, 2)

plt.plot(train_steps, smoothed_train_t4_lpips, label='App1 t4 LPIPS (Smoothed)')
plt.plot(o_train_steps, smoothed_o_train_t4_lpips, label='App2 t4 LPIPS (Smoothed)')

plt.plot(t_train_steps, smoothed_t_train_t4_lpips, label='Original t4 LPIPS (Smoothed)')
plt.xlabel('Training Step')
plt.ylabel('LPIPS')
plt.title(f'Training LPIPS for t1, t3, t4 (Window Size: {window_size_2})')
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(val_steps, smoothed_val_psnr, label='App1 Validation PSNR (Smoothed)')
plt.plot(o_val_steps, smoothed_o_val_psnr, label='App2 Validation PSNR (Smoothed)')

plt.plot(t_val_steps, smoothed_t_val_psnr, label='Original Validation PSNR (Smoothed)')
plt.xlabel('Validation Step')
plt.ylabel('PSNR')
plt.title(f'Validation PSNR over Steps (Window Size: {window_size_1})')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_steps, smoothed_val_lpips, label='App1 Validation LPIPS (Smoothed)', color='orange')
plt.plot(t_val_steps, smoothed_t_val_lpips, label='Original Validation LPIPS (Smoothed)', color='red')
plt.plot(t_val_steps, smoothed_o_val_lpips, label='App2 Validation LPIPS (Smoothed)', color='blue')
plt.xlabel('Validation Step')
plt.ylabel('LPIPS')
plt.title(f'Validation LPIPS over Steps (Window Size: {window_size_2})')
plt.legend()

plt.tight_layout()
plt.show()