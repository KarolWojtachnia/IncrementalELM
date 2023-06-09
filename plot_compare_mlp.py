import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def scores_to_cummean(scores):
    return np.cumsum(scores)/np.arange(1,len(scores)+1)

labels = ['ELM','MLP','Ensemble']
scores = np.load('res/scores_vs.npy')
ttimes = np.load('res/ttimes_vs.npy')
ptimes = np.load('res/ptimes_vs.npy')

scores = np.mean(scores, axis=0)
ttimes = np.mean(ttimes, axis=0)
ptimes = np.mean(ptimes, axis=0)

cols = ['cornflowerblue', 'tomato', 'purple']
k=1

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for m_id, m_label in enumerate(labels):
    ax[0].plot(gaussian_filter(scores[m_id], k), label=m_label, c=cols[m_id], alpha=0.5)
    ax[1].plot(scores_to_cummean(scores[m_id]), label=m_label, c=cols[m_id], alpha=0.5)
    ax[2].plot(scores_to_cummean(ptimes[m_id]), label=m_label, c=cols[m_id], alpha=0.5)

ax[0].set_ylabel('accuracy')
ax[2].set_xlabel('chunk')
ax[1].set_ylabel('accumulated accuracy')
ax[2].set_ylabel('accumulated training time')

ax[0].set_ylim(0.5,0.9)
ax[1].set_ylim(0.5,0.8)
ax[2].set_xlim(0,249)
ax[0].legend(frameon=False, ncol=3)

for a in ax:
    a.grid(ls=':')
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/compare.png')