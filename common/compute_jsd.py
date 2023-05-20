
import numpy as np
from scipy.spatial.distance import jensenshannon    


def compute_jsd_from_counters(p_counter, q_counter):
    # measure Jenson-Shannon distance for shared vocab between two sets of substitute terms    
    vocab = sorted(set(p_counter).union(q_counter))
    p_counts = np.array([p_counter[v] for v in vocab])
    q_counts = np.array([q_counter[v] for v in vocab])
    p_dist = p_counts / p_counts.sum()
    q_dist = q_counts / q_counts.sum()
    return jensenshannon(p_dist, q_dist, base=2)
    
