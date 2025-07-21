import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

sigma = 1 if len(train_losses) >= 10 else 0
train_losses_smooth = gaussian_filter1d(train_losses, sigma=sigma)
val_losses_smooth = gaussian_filter1d(val_losses, sigma=sigma)

best_epoch = np.argmin(val_losses)
best_loss = val_losses[best_epoch]

sns.set(style="whitegrid", context='talk')

plt.figure(figsize=(12, 7))
plt.plot(train_losses_smooth, label='Train Loss', color='steelblue', linewidth=2.5)
plt.plot(val_losses_smooth, label='Validation Loss', color='darkorange', linewidth=2.5)

plt.scatter(best_epoch, best_loss, color='red', s=100, zorder=5, label=f'Best Epoch ({best_epoch})')

y_offset = (val_losses.max() - val_losses.min()) * 0.05
plt.annotate(f'{best_loss:.2f}',
             xy=(best_epoch, best_loss),
             xytext=(best_epoch + 3, best_loss + y_offset),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
             fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

plt.title('VAE Loss Curve During Training (HYDRA)', fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)

plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('vae_loss_curve_a_star.png', dpi=300, bbox_inches='tight')
plt.show()