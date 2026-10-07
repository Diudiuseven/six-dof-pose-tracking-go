package tracking

import (
	"sort"
	"sync"
)

// event is a timestamped sensor reading on the unified timeline.
type event struct {
	t        int64 // unified ns
	isVision bool
	imu      IMUSample
	vis      VisionObservation
}

// checkpoint captures everything needed to roll the estimator back
// to a past instant for out-of-order reprocessing.
type checkpoint struct {
	t               int64
	state           filterState
	lastFusedVision int64
	trackState      TrackingState
	blend           recoveryBlend
}

// recoveryBlend describes the smooth transition back to the fused
// estimate after re-acquisition.
type recoveryBlend struct {
	active   bool
	start    int64 // unified ns
	deltaPos Vec3  // world-frame offset, decays to zero
	deltaQ   Quat  // world-frame rotation offset, decays to identity
}

// Tracker is the top-level runtime. It is safe for concurrent use:
// ingestion methods may run on sensor goroutines while PoseAt is
// called from a render/query thread. Queries take a read lock and
// never block writers for more than one event-processing step.
type Tracker struct {
	cfg Config

	mu         sync.RWMutex
	imuClock   *clockAligner
	visClock   *clockAligner
	filt       *filter
	frontier   int64 // unified ns of the newest processed event; 0 = none
	lastFused  int64 // unified ns of last accepted vision; 0 = none
	trackState TrackingState
	blend      recoveryBlend

	events  []event      // reprocessing window, sorted by t
	cps     []checkpoint // one per processed event, aligned with events
	seenIMU map[int64]struct{}
	seenVis map[int64]struct{}

	out     []PoseSnapshot // output history, sorted by t
	seq     uint64
	rejects []RejectEvent
}

// NewTracker creates a tracker with the given configuration.
func NewTracker(cfg Config) *Tracker {
	c := cfg.withDefaults()
	return &Tracker{
		cfg:        c,
		imuClock:   newClockAligner(),
		visClock:   newClockAligner(),
		filt:       newFilter(c),
		seenIMU:    make(map[int64]struct{}),
		seenVis:    make(map[int64]struct{}),
		trackState: StateInitializing,
	}
}

// IngestIMU feeds one IMU sample. unifiedNow is the current time on
// the unified timeline (ns), used for clock alignment.
func (t *Tracker) IngestIMU(s IMUSample, unifiedNow int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.imuClock.observe(s.Timestamp, unifiedNow)
	tu := t.imuClock.toUnified(s.Timestamp)
	if !isFiniteVec(s.Accel) || !isFiniteVec(s.Gyro) {
		t.logReject(tu, "imu", RejectNonFinite)
		return
	}
	t.ingestLocked(event{t: tu, imu: s}, t.seenIMU, "imu", t.cfg.MaxLateIMU.Nanoseconds())
}

// IngestVision feeds one vision observation. It returns the reject
// reason (RejectNone if fused).
func (t *Tracker) IngestVision(o VisionObservation, unifiedNow int64) RejectReason {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.visClock.observe(o.Timestamp, unifiedNow)
	tu := t.visClock.toUnified(o.Timestamp)
	if !isFiniteVec(o.Pose.Position) || !isFiniteQuat(o.Pose.Orientation) {
		t.logReject(tu, "vision", RejectNonFinite)
		return RejectNonFinite
	}
	o.Timestamp = tu
	return t.ingestLocked(event{t: tu, isVision: true, vis: o}, t.seenVis, "vision",
		t.cfg.MaxLateVision.Nanoseconds())
}

// ingestLocked inserts an event into the ordered log and processes
// it, rolling back and replaying if it arrived out of order.
func (t *Tracker) ingestLocked(e event, seen map[int64]struct{}, src string, maxLate int64) RejectReason {
	if _, dup := seen[e.t]; dup {
		t.logReject(e.t, src, RejectDuplicate)
		return RejectDuplicate
	}
	if t.frontier != 0 && e.t < t.frontier-maxLate {
		t.logReject(e.t, src, RejectTooLate)
		return RejectTooLate
	}
	seen[e.t] = struct{}{}

	// Find insertion point (first index with events[i].t > e.t).
	idx := sort.Search(len(t.events), func(i int) bool { return t.events[i].t > e.t })
	t.events = append(t.events, event{})
	copy(t.events[idx+1:], t.events[idx:])
	t.events[idx] = e

	var reason RejectReason
	if idx == len(t.events)-1 {
		// In-order: just process at the frontier.
		reason = t.processEvent(e)
		t.cps = append(t.cps, t.snapshotCheckpoint(e.t))
	} else {
		// Out-of-order: roll back to the checkpoint before idx and
		// replay everything from there.
		reason = t.rollbackAndReplay(idx)
	}
	t.pruneLocked()
	return reason
}

// processEvent applies one event to the filter at the frontier.
// It returns the fusion result for vision observations.
func (t *Tracker) processEvent(e event) RejectReason {
	reason := RejectNone
	if e.isVision {
		// Capture the output pose implied by the pre-update
		// (inertial-propagated) state so a recovery blend can start
		// from exactly what was last emitted.
		preOutput := t.outputPoseAtLocked(e.t)
		reason = t.filt.updateVision(e.vis)
		if reason != RejectNone {
			t.logReject(e.t, "vision", reason)
		} else {
			t.onVisionAccepted(e.t, preOutput)
		}
	} else {
		dt := 0.0
		if t.frontier != 0 && e.t > t.frontier {
			dt = float64(e.t-t.frontier) * 1e-9
		}
		s := e.imu
		s.Timestamp = e.t
		t.filt.predict(s, dt)
	}
	if e.t > t.frontier {
		t.frontier = e.t
	}
	// Numerical safety: never emit a diverged estimate.
	if t.filt.diverged() {
		p := t.filt.st.pose
		if !isFiniteVec(p.Position) || !isFiniteQuat(p.Orientation) {
			p = Pose{Orientation: Quat{W: 1}}
		}
		t.filt.reset(p)
		t.trackState = StateLost
	}
	t.updateStateLocked()
	t.appendOutputLocked(e.t)
	return reason
}

// onVisionAccepted updates tracking state and, when coming back from
// a non-tracking state, starts a smooth recovery blend so the output
// pose does not jump.
func (t *Tracker) onVisionAccepted(ts int64, before Pose) {
	wasTracking := t.trackState == StateTracking || t.trackState == StateRecovering
	t.lastFused = ts
	if !wasTracking {
		after := t.filt.st.pose
		t.blend = recoveryBlend{
			active:   true,
			start:    ts,
			deltaPos: vSub(before.Position, after.Position),
			deltaQ:   qMul(before.Orientation, qConj(after.Orientation)),
		}
	}
}

// updateStateLocked recomputes the tracking state from timing.
func (t *Tracker) updateStateLocked() {
	switch {
	case t.lastFused == 0:
		t.trackState = StateInitializing
	case t.frontier-t.lastFused > t.cfg.VisionTimeout.Nanoseconds():
		t.trackState = StateLost
	case t.blend.active:
		t.trackState = StateRecovering
	default:
		t.trackState = StateTracking
	}
}

// blendFactor returns how much of the recovery offset remains at
// time ts (1 at blend start, 0 once fully blended).
func (t *Tracker) blendFactor(ts int64) float64 {
	if !t.blend.active {
		return 0
	}
	d := t.cfg.RecoveryBlend.Nanoseconds()
	rem := 1 - float64(ts-t.blend.start)/float64(d)
	if rem <= 0 {
		t.blend.active = false
		return 0
	}
	if rem > 1 {
		return 1
	}
	return rem
}

// outputPoseAtLocked computes the blended output pose at time ts
// from the current filter state. Only valid at the frontier.
func (t *Tracker) outputPoseAtLocked(ts int64) Pose {
	p := t.filt.st.pose
	k := t.blendFactor(ts)
	if k > 0 {
		p.Position = vAdd(p.Position, vScale(t.blend.deltaPos, k))
		p.Orientation = qNormalize(qMul(qSlerp(Quat{W: 1}, t.blend.deltaQ, k), p.Orientation))
	}
	return p
}

// appendOutputLocked records the output snapshot at the frontier.
func (t *Tracker) appendOutputLocked(ts int64) {
	t.seq++
	snap := PoseSnapshot{
		Timestamp:       ts,
		Pose:            t.outputPoseAtLocked(ts),
		Velocity:        t.filt.st.vel,
		State:           t.trackState,
		CovarianceTrace: trace3(t.filt.st.cov, errPos),
		Sequence:        t.seq,
	}
	// Replace an existing entry at the same timestamp (replay case).
	if n := len(t.out); n > 0 && t.out[n-1].Timestamp == ts {
		t.out[n-1] = snap
		return
	}
	t.out = append(t.out, snap)
	if len(t.out) > t.cfg.HistorySize {
		t.out = append([]PoseSnapshot(nil), t.out[len(t.out)-t.cfg.HistorySize:]...)
	}
}

func (t *Tracker) snapshotCheckpoint(ts int64) checkpoint {
	return checkpoint{
		t:               ts,
		state:           t.filt.st,
		lastFusedVision: t.lastFused,
		trackState:      t.trackState,
		blend:           t.blend,
	}
}

// rollbackAndReplay restores the checkpoint just before index idx
// and reprocesses all events from idx onward.
func (t *Tracker) rollbackAndReplay(start int) RejectReason {
	// Restore state from the checkpoint before the insertion point.
	if start > 0 {
		cp := t.cps[start-1]
		t.filt.st = cp.state
		t.lastFused = cp.lastFusedVision
		t.trackState = cp.trackState
		t.blend = cp.blend
		t.frontier = cp.t
	} else {
		t.filt.reset(Pose{Orientation: Quat{W: 1}})
		t.lastFused = 0
		t.trackState = StateInitializing
		t.blend = recoveryBlend{}
		t.frontier = 0
	}
	// Truncate checkpoints, outputs and rejects past the rollback so
	// replayed events do not double-log.
	t.cps = t.cps[:start]
	cut := t.cpsFrontier(start)
	t.out = t.out[:cut]
	ri := sort.Search(len(t.rejects), func(i int) bool { return t.rejects[i].Timestamp > t.frontier })
	t.rejects = t.rejects[:ri]

	var reason RejectReason
	for i := start; i < len(t.events); i++ {
		r := t.processEvent(t.events[i])
		if i == start {
			reason = r
		}
		t.cps = append(t.cps, t.snapshotCheckpoint(t.events[i].t))
	}
	return reason
}

// cpsFrontier returns the output-history length covering snapshots
// up to checkpoint index start.
func (t *Tracker) cpsFrontier(start int) int {
	if start == 0 {
		return 0
	}
	ts := t.cps[start-1].t
	i := sort.Search(len(t.out), func(i int) bool { return t.out[i].Timestamp > ts })
	return i
}

// pruneLocked drops events and state outside the reprocessing window.
func (t *Tracker) pruneLocked() {
	if t.frontier == 0 {
		return
	}
	window := t.cfg.MaxLateVision.Nanoseconds() + t.cfg.MaxLateIMU.Nanoseconds() + int64(10e6)
	cutoff := t.frontier - window
	i := 0
	for i < len(t.events) && t.events[i].t < cutoff {
		if t.events[i].isVision {
			delete(t.seenVis, t.events[i].t)
		} else {
			delete(t.seenIMU, t.events[i].t)
		}
		i++
	}
	if i > 0 {
		t.events = append([]event(nil), t.events[i:]...)
		t.cps = append([]checkpoint(nil), t.cps[i:]...)
	}
}

// PoseAt returns a self-consistent pose snapshot interpolated (or
// extrapolated within limits) to the requested unified time.
func (t *Tracker) PoseAt(unifiedTime int64) PoseSnapshot {
	t.mu.RLock()
	defer t.mu.RUnlock()
	n := len(t.out)
	if n == 0 {
		return PoseSnapshot{Timestamp: unifiedTime, State: t.trackState,
			Pose: Pose{Orientation: Quat{W: 1}}}
	}
	// Find first snapshot with Timestamp > unifiedTime.
	i := sort.Search(n, func(i int) bool { return t.out[i].Timestamp > unifiedTime })
	switch {
	case i == 0:
		s := t.out[0]
		s.Timestamp = unifiedTime
		return s
	case i == n:
		// Extrapolate forward, capped at 100ms.
		s := t.out[n-1]
		dt := float64(unifiedTime-s.Timestamp) * 1e-9
		if dt > 0.1 {
			dt = 0.1
		}
		if dt < 0 {
			dt = 0
		}
		s.Pose.Position = vAdd(s.Pose.Position, vScale(s.Velocity, dt))
		s.Timestamp = unifiedTime
		return s
	default:
		a, b := t.out[i-1], t.out[i]
		span := b.Timestamp - a.Timestamp
		f := float64(0)
		if span > 0 {
			f = float64(unifiedTime-a.Timestamp) / float64(span)
		}
		return PoseSnapshot{
			Timestamp: unifiedTime,
			Pose: Pose{
				Position:    vAdd(a.Pose.Position, vScale(vSub(b.Pose.Position, a.Pose.Position), f)),
				Orientation: qSlerp(a.Pose.Orientation, b.Pose.Orientation, f),
			},
			Velocity:        vAdd(a.Velocity, vScale(vSub(b.Velocity, a.Velocity), f)),
			State:           b.State,
			CovarianceTrace: a.CovarianceTrace + (b.CovarianceTrace-a.CovarianceTrace)*f,
			Sequence:        b.Sequence,
		}
	}
}

// State returns the current tracking state.
func (t *Tracker) State() TrackingState {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.trackState
}

// Rejects returns a copy of the diagnostic reject log.
func (t *Tracker) Rejects() []RejectEvent {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return append([]RejectEvent(nil), t.rejects...)
}

func (t *Tracker) logReject(ts int64, src string, r RejectReason) {
	t.rejects = append(t.rejects, RejectEvent{Timestamp: ts, Source: src, Reason: r})
	const maxRejects = 1024
	if len(t.rejects) > maxRejects {
		t.rejects = append([]RejectEvent(nil), t.rejects[len(t.rejects)-maxRejects:]...)
	}
}
