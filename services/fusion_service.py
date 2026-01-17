def fuse(identity_score, liveness_score,
         id_threshold=0.65,
         live_threshold=0.7):
    """
    Final decision fusion.
    
    Identity and liveness are treated as independent security checks.
    Access is granted only if both exceed their respective thresholds.
    """

    identity_pass = identity_score >= id_threshold
    liveness_pass = liveness_score >= live_threshold

    return identity_pass and liveness_pass