from .sample import Sample

def split_holdout(rows, n_holdout, seed):
    """Shuffle deterministically and carve off a disjoint held-out prefix.
    Returns (heldout_rows, train_rows), both reindexed to sequential ids."""
    import random
    rows=list(rows); order=list(range(len(rows))); random.Random(seed).shuffle(order)
    ordered=[rows[i] for i in order]
    heldout,train=ordered[:n_holdout],ordered[n_holdout:]
    return reindex(heldout),reindex(train)

def reindex(rows):
    return [Sample(str(i),r.messages,r.noise,r.meta) for i,r in enumerate(rows)]

def split_fractions(rows, fractions, seed):
    """Shuffle deterministically and cut into len(fractions) disjoint pieces by cumulative
    proportion (remainder goes to the last piece). Returns a list of reindexed row-lists,
    same order as `fractions`."""
    import random
    rows=list(rows); order=list(range(len(rows))); random.Random(seed).shuffle(order)
    ordered=[rows[i] for i in order]
    n=len(ordered); cuts=[0]
    acc=0.0
    for frac in fractions[:-1]:
        acc+=frac; cuts.append(round(n*acc))
    cuts.append(n)
    return [reindex(ordered[cuts[i]:cuts[i+1]]) for i in range(len(fractions))]
