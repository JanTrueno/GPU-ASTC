"""ASTC 2-partition patterns for 4x4 blocks and the nearest-pattern table the shader uses.
lut()[mask] = seed | distance << 10 | pattern << 16, where bit i of mask/pattern is the group of texel i (x + 4*y)."""
import numpy as np

M = 0xFFFFFFFF

def hash52(p):
    p ^= p >> 15; p = (p - (p << 17)) & M; p = (p + (p << 7)) & M; p = (p + (p << 4)) & M
    p ^= p >> 5; p = (p + (p << 16)) & M; p ^= p >> 7; p ^= p >> 3
    p = (p ^ (p << 6)) & M; p ^= p >> 17
    return p

def select_partition(seed, x, y, count=2, small=True):
    if small: x <<= 1; y <<= 1
    seed += (count - 1) * 1024
    r = hash52(seed)
    s = [(r >> k) & 0xF for k in (0, 4, 8, 12, 16, 20, 24, 28, 18, 22, 26)] + [((r >> 30) | (r << 2)) & 0xF]
    s = [v * v for v in s]
    if seed & 1: sh1 = 4 if seed & 2 else 5; sh2 = 6 if count == 3 else 5
    else: sh1 = 6 if count == 3 else 5; sh2 = 4 if seed & 2 else 5
    sh3 = sh1 if seed & 0x10 else sh2
    for k in range(8): s[k] >>= sh1 if k % 2 == 0 else sh2
    for k in range(8, 12): s[k] >>= sh3
    a = (s[0] * x + s[1] * y + (r >> 14)) & 0x3F
    b = (s[2] * x + s[3] * y + (r >> 10)) & 0x3F
    c = (s[4] * x + s[5] * y + (r >> 6)) & 0x3F if count >= 3 else 0
    d = (s[6] * x + s[7] * y + (r >> 2)) & 0x3F if count >= 4 else 0
    if a >= b and a >= c and a >= d: return 0
    if b >= c and b >= d: return 1
    return 2 if c >= d else 3

def patterns():
    return [sum(select_partition(seed, i & 3, i >> 2) << i for i in range(16)) for seed in range(1024)]

def lut():
    dist = np.full(65536, 99, np.int32); val = np.zeros(65536, np.uint32)
    for seed, p in reversed(list(enumerate(patterns()))):   # lowest seed wins
        if p in (0, 0xFFFF): continue
        for m in (p, p ^ 0xFFFF): dist[m] = 0; val[m] = seed | (p << 16)
    idx = np.arange(65536)
    changed = True
    while changed:
        changed = False
        for b in range(16):
            nb = idx ^ (1 << b); better = dist[nb] + 1 < dist
            if better.any(): dist[better] = dist[nb][better] + 1; val[better] = val[nb][better]; changed = True
    return val | (dist.astype(np.uint32) << 10)

# ---------------------------------------------------------------------------
# 5x5 and 6x6: a mask no longer fits a table index. The table is keyed by a 16-texel sample of
# the wanted split and names the three patterns closest on that sample; the shader compares
# those three in full. Masks hold rows 0-2 in the first word and the remaining rows in the
# second, dim bits per row.
# big_table(dim): 264 rows of 256 uint32. Rows 0-255: seed0 | seed1 << 10 | seed2 << 20 by key.
# Rows 256-263: the two mask words of every seed.
# ---------------------------------------------------------------------------
SAMPLE = {5: (0, 1, 3, 4), 6: (0, 2, 3, 5)}

def big_masks(dim):
    small = dim * dim < 31
    out = []
    for seed in range(1024):
        lo = hi = 0
        for y in range(dim):
            for x in range(dim):
                if select_partition(seed, x, y, 2, small):
                    if y < 3: lo |= 1 << (dim * y + x)
                    else: hi |= 1 << (dim * (y - 3) + x)
        out.append((lo, hi))
    return out

def sample_key(dim, lo, hi):
    key = 0
    for r, y in enumerate(SAMPLE[dim]):
        row = (lo >> (dim * y)) if y < 3 else (hi >> (dim * (y - 3)))
        for c, x in enumerate(SAMPLE[dim]):
            key |= ((row >> x) & 1) << (4 * r + c)
    return key

def big_table(dim):
    masks = big_masks(dim)
    all_lo = (1 << (3 * dim)) - 1; all_hi = (1 << ((dim - 3) * dim)) - 1
    seen = set(); seeds = []
    for seed, (lo, hi) in enumerate(masks):          # lowest seed of each distinct split
        if (lo, hi) in ((0, 0), (all_lo, all_hi)): continue
        canon = min((lo, hi), (lo ^ all_lo, hi ^ all_hi))
        if canon in seen: continue
        seen.add(canon); seeds.append(seed)
    sig = np.array([sample_key(dim, *masks[s]) for s in seeds], np.uint16)
    pop = np.array([bin(i).count("1") for i in range(65536)], np.int32)
    table = np.zeros((264, 256), np.uint32)
    rank = np.arange(len(seeds), dtype=np.int32)
    for k0 in range(0, 65536, 4096):
        keys = np.arange(k0, k0 + 4096, dtype=np.uint16)
        d = pop[keys[:, None] ^ sig[None, :]]; d = np.minimum(d, 16 - d)
        order = d * 2048 + rank[None, :]             # nearest first, lowest seed on ties
        top = np.sort(np.partition(order, 2, axis=1)[:, :3], axis=1) & 2047
        s3 = np.array(seeds, np.uint32)[top]
        table.reshape(-1)[k0:k0 + 4096] = s3[:, 0] | (s3[:, 1] << 10) | (s3[:, 2] << 20)
    flat = table.reshape(-1)
    for seed, (lo, hi) in enumerate(masks):
        flat[65536 + 2 * seed] = lo; flat[65536 + 2 * seed + 1] = hi
    return table

if __name__ == "__main__":
    v = lut(); print("unique patterns:", len(set(int(x) >> 16 for x in v)))
