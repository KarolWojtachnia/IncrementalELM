import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def find_real_drift(chunks, drifts):
    interval = (chunks/drifts)
    idx = [interval*(i+.5) for i in range(drifts)]
    return np.array(idx).astype(int)

n_drifts = 18
n_chunks = 250
prev = 6
post = 12

drifts = find_real_drift(n_chunks, n_drifts)
 
######

def scores_to_cummean(scores):
    return np.cumsum(scores)/np.arange(1,len(scores)+1)

labels = ['ELM','MLP','Ensemble']
scores = np.load('res/scores_vs.npy')
scores = np.mean(scores, axis=0)

scores_r = np.array([scores[:,d-prev:d+post] for d in drifts[:-1]])
scores_r_m = np.mean(scores_r, axis=0)

cols = ['cornflowerblue', 'tomato', 'purple']
k=1

fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), sharex=True)
ax.set_title('Average restoration curve')
for m_id, m_label in enumerate(labels):
    ax.plot(gaussian_filter(scores_r_m[m_id], k), label=m_label, c=cols[m_id], alpha=0.75)

ax.vlines(prev, 0.5, 0.9, color='black', alpha=0.5, ls=':')

ax.set_ylabel('accuracy')
ax.set_xlabel('chunk')

ax.set_ylim(0.5,0.9)
ax.set_xlim(0, prev+post-1)
ax.set_xticks(np.arange(0, prev+post-1)[::2])
ax.legend(frameon=False, ncol=3)

ax.grid(ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/zoom_avg.png')