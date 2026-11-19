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

if __name__ == "__main__":
    v = lut(); print("unique patterns:", len(set(int(x) >> 16 for x in v)))
