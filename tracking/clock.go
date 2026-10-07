package tracking

// clockAligner maps timestamps from a sensor clock domain onto the
// unified tracker timeline. The offset between the two clocks is
// estimated with a bounded exponential moving average over observed
// (sensorTime, unifiedTime) pairs, which tolerates slow drift while
// remaining fully deterministic (no randomness, no wall-clock reads).
//
// The first observed pair fixes the offset; subsequent pairs update it
// with a small gain so a single jittery arrival cannot swing the
// timeline.
type clockAligner struct {
	offset  int64 // unified = sensor + offset
	seen    bool
	samples uint64
}

func newClockAligner() *clockAligner { return &clockAligner{} }

// maxOffsetCorrection caps per-sample offset adjustment to avoid
// timeline jumps when a burst of delayed samples arrives at once.
const maxOffsetCorrection = int64(5e6) // 5ms

// observe records a (sensorTime, unifiedTime) pair.
func (c *clockAligner) observe(sensorTime, unifiedTime int64) {
	cand := unifiedTime - sensorTime
	if !c.seen {
		c.offset = cand
		c.seen = true
		c.samples = 1
		return
	}
	diff := cand - c.offset
	// Gain: 1/8 for early convergence, then 1/64 for stability.
	shift := uint(6)
	if c.samples < 64 {
		shift = 3
	}
	corr := diff >> shift
	if corr > maxOffsetCorrection {
		corr = maxOffsetCorrection
	} else if corr < -maxOffsetCorrection {
		corr = -maxOffsetCorrection
	}
	c.offset += corr
	c.samples++
}

// toUnified converts a sensor timestamp to the unified timeline.
// Before any observation it is the identity mapping.
func (c *clockAligner) toUnified(sensorTime int64) int64 {
	if !c.seen {
		return sensorTime
	}
	return sensorTime + c.offset
}
