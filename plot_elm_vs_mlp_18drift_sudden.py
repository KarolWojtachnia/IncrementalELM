import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def scores_to_cummean(scores):
    return np.cumsum(scores)/np.arange(1,len(scores)+1)

labels = ['ELM','MLP','Ensemble']
scores = np.load('res/elm_vs_mlp_18drift_sudden.npy')

scores = np.mean(scores, axis=0)

cols = ['cornflowerblue', 'tomato', 'purple']
k=1

fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

for m_id, m_label in enumerate(labels):
    ax[0].plot(gaussian_filter(scores[m_id], k), label=m_label, c=cols[m_id], alpha=0.5)
    ax[1].plot(scores_to_cummean(scores[m_id]), label=m_label, c=cols[m_id], alpha=0.5)

ax[0].set_ylabel('accuracy')
ax[1].set_xlabel('chunk')
ax[1].set_ylabel('accumulated accuracy')

ax[0].set_ylim(0.5,0.9)
ax[1].set_ylim(0.5,0.8)
ax[1].set_xlim(0,249)
ax[0].legend(frameon=False, ncol=3)

for a in ax:
    a.grid(ls=':')
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/elm_vs_mlp_18drift_sudden.png')