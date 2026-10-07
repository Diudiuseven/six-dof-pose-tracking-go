package tracking

import "sort"

// buffer.go: time-ordered event logs that absorb out-of-order, late and
// duplicate arrivals. Events are keyed by unified timestamp; insertion
// is by binary search so a late event lands at its correct position.
// Logs are pruned to the configured history window; anything older is
// rejected by the caller before insertion.

// imuEntry is an accepted IMU sample on the unified timeline.
type imuEntry struct {
	Time  int64
	Seq   uint64
	Accel Vector3
	Gyro  Vector3
}

// visionEntry is an accepted vision observation on the unified timeline.
// committed is false while the observation still has to pass the outlier
// gate during the next replay.
type visionEntry struct {
	Time       int64
	Seq        uint64
	Pose       Pose
	Covariance float64
	committed  bool
}

// imuLog is a sorted, bounded log of IMU entries.
type imuLog struct {
	entries []imuEntry
}

// insert adds e in time order and returns false if an entry with the
// same timestamp already exists (exact-timestamp collision).
func (l *imuLog) insert(e imuEntry) bool {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time >= e.Time })
	if i < len(l.entries) && l.entries[i].Time == e.Time {
		return false
	}
	l.entries = append(l.entries, imuEntry{})
	copy(l.entries[i+1:], l.entries[i:])
	l.entries[i] = e
	return true
}

// prune drops entries at or before t.
func (l *imuLog) prune(t int64) {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time > t })
	l.entries = append([]imuEntry(nil), l.entries[i:]...)
}

// from returns the entries with Time > t.
func (l *imuLog) from(t int64) []imuEntry {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time > t })
	return l.entries[i:]
}

// visionLog is a sorted, bounded log of vision entries.
type visionLog struct {
	entries []visionEntry
}

func (l *visionLog) insert(e visionEntry) bool {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time >= e.Time })
	if i < len(l.entries) && l.entries[i].Time == e.Time {
		return false
	}
	l.entries = append(l.entries, visionEntry{})
	copy(l.entries[i+1:], l.entries[i:])
	l.entries[i] = e
	return true
}

func (l *visionLog) prune(t int64) {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time > t })
	l.entries = append([]visionEntry(nil), l.entries[i:]...)
}

// from returns the entries with Time > t.
func (l *visionLog) from(t int64) []visionEntry {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time > t })
	return l.entries[i:]
}

// seqWindow tracks recently consumed sequence numbers of one stream for
// duplicate detection. It is exact for the last `size` distinct
// sequence numbers and conservative beyond that.
type seqWindow struct {
	seen map[uint64]struct{}
	// order holds seqs in arrival order so the window can slide.
	order []uint64
	size  int
}

func newSeqWindow(size int) *seqWindow {
	return &seqWindow{seen: make(map[uint64]struct{}), size: size}
}

// add returns false if seq was already present.
func (w *seqWindow) add(seq uint64) bool {
	if _, dup := w.seen[seq]; dup {
		return false
	}
	w.seen[seq] = struct{}{}
	w.order = append(w.order, seq)
	for len(w.order) > w.size {
		delete(w.seen, w.order[0])
		w.order = w.order[1:]
	}
	return true
}

// removeAt deletes the entry with the given timestamp, if present.
func (l *visionLog) removeAt(t int64) {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time >= t })
	if i < len(l.entries) && l.entries[i].Time == t {
		l.entries = append(l.entries[:i], l.entries[i+1:]...)
	}
}

// markCommitted flags the entry with the given timestamp as committed.
func (l *visionLog) markCommitted(t int64) {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time >= t })
	if i < len(l.entries) && l.entries[i].Time == t {
		l.entries[i].committed = true
	}
}

// has reports whether an entry with the given timestamp exists.
func (l *visionLog) has(t int64) bool {
	i := sort.Search(len(l.entries), func(i int) bool { return l.entries[i].Time >= t })
	return i < len(l.entries) && l.entries[i].Time == t
}
