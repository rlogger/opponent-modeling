# Representative objective trajectories

![Synchronized prey-seeking, risk-averse, and curious trajectories](all-trajectories.gif)

These animations show one deterministic matched reset from the regenerated
1,800-episode dataset. All three specialists use the same checkpoint seed,
environment key, initial scene, and fixed capture-prey control, so their paths
are directly comparable.

## Individual trajectories

### Prey-seeking

![Prey-seeking trajectory](prey-seeking.gif)

### Risk-averse

![Risk-averse trajectory](risk-averse.gif)

### Curious

![Curious trajectory](curious.gif)

## Static plot

![Representative trajectory triptych](representative-trajectories.png)

[Download the vector PDF](representative-trajectories.pdf).

The GIFs share a 100-step clock. Paths reveal only their valid prefix: the
prey-seeking example captures at step 31 and then freezes, while the risk-averse
and curious examples continue to the horizon. See `animation-manifest.json` for
the source hash, selected episode indices, timing, dimensions, and output
hashes.

These are behavior rollouts from trained specialists. They do not by themselves
establish representation recovery or downstream scientific success.
