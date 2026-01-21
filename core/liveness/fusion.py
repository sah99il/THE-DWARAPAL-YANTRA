def fuse_scores(texture, temporal, rppg,
                w1=0.3, w2=0.3, w3=0.4):
    score = w1 * texture + w2 * temporal + w3 * rppg
    return float(min(max(score, 0.0), 1.0))
