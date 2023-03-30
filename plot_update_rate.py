import matplotlib.pyplot as plt
import numpy as np

def scores_to_cummean(scores):
    return np.cumsum(scores)/np.arange(1,len(scores)+1)

deltas = np.linspace(0.1, 1, 10)
res = np.load('res/update_rate.npy')
res = np.mean(res, axis=0)

res_c = np.zeros_like(res)

for r_id, r in enumerate(res):
    res_c[r_id] = scores_to_cummean(r)
    
print(res.shape)

cols = plt.cm.jet(np.linspace(0,1,len(deltas)))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))


ax.imshow(res_c[:,10:], cmap='coolwarm', aspect='auto')

ax.set_yticks(np.arange(len(deltas)), ['%.1f' % d for d in deltas])

ax.set_title('Update rate comparison | accumulated accuracy')
ax.set_xlabel('Chunk number')
ax.set_ylabel('update rate')  


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/update_rate.png')
