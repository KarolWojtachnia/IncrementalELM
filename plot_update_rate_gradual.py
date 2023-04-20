import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def scores_to_cummean(scores):
    return np.cumsum(scores)/np.arange(1,len(scores)+1)

labels = ['0.2', '0.5','0.8']
scores = np.load('res/update_rate_gradual.npy')

scores = np.mean(scores, axis=0)

cols = ['cornflowerblue', 'tomato', 'purple']
k=1

fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

for m_id, m_label in enumerate(labels):
    ax.plot(gaussian_filter(scores[m_id], k), label=m_label, c=cols[m_id], alpha=0.5)

ax.set_ylabel('accuracy')
ax.set_xlabel('chunk')

ax.set_xlim(1, 250)
ax.set_ylim(0.5,0.7)
ax.legend(frameon=False, ncol=3)

ax.grid(ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.title('Accuracy over the whole stream for gradual drift')
plt.savefig('foo.png')
plt.savefig('fig/compare_update_gradual.png')
