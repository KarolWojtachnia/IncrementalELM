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

fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), sharex=True)

for m_id, m_label in enumerate(labels):
    ax.plot(gaussian_filter(scores[m_id], k), label=m_label, c=cols[m_id], alpha=0.75)

ax.set_ylabel('accuracy')
ax.set_xlabel('chunk')

ax.set_ylim(0.5,0.9)
ax.set_xlim(15, 30)
ax.legend(frameon=False, ncol=3)

ax.grid(ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/zoom.png')