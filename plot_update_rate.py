import matplotlib.pyplot as plt
import numpy as np

def scores_to_cummean(scores):
    return np.cumsum(scores)/np.arange(1,len(scores)+1)

deltas = np.linspace(0.1, 1, 10)
res = np.load('res/update_rate.npy')
res = np.mean(res, axis=0)
print(res.shape)

cols = plt.cm.jet(np.linspace(0,1,len(deltas)))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for r_id, r in enumerate(deltas):
    t = scores_to_cummean(res[r_id])
    ax.plot(t, label="update rate %.1f" % r, c=cols[r_id], alpha=0.5)

ax.legend()
ax.grid(ls=":")
ax.set_ylim(0.62, 0.67)
ax.set_title('Update rate comparison')
ax.set_xlabel('Chunk number')
ax.set_ylabel('accuracy')    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/update_rate.png')
