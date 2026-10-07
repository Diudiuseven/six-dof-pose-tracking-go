package tracking

import (
	"sort"
	"sync"
)

// tracker.go: public API and orchestration.
//
// Internally the tracker keeps accepted IMU and vision events in
// time-ordered logs inside a bounded history window, plus checkpoints of
// the full fusion state at every accepted vision event. Late arrivals
// are inserted at their correct position and the affected suffix of the
// timeline is deterministically replayed from the nearest checkpoint,
// so a late observation lands exactly as if it had arrived on time.
//
// Readers only ever see the snapshot ring: immutable PoseSample values
// published after each fused step. A query therefore always returns one
// self-consistent snapshot (or an interpolation between two), never a
// field-level mix, and writers never block on readers beyond the short
// critical section that appends to the ring.

// maxSnapshots bounds the published snapshot ring.
const maxSnapshots = 8192

// checkpoint is the complete internal state at a point on the unified
// timeline. Replaying the event logs from any checkpoint reproduces the
// exact same downstream states.
type checkpoint struct {
	est            estimator
	haveVision     bool // an absolute observation was accepted at least once
	offPos         Vector3
	offRot         Vector3 // output smoothing offset (rotation vector)
	recStart       int64   // start of the current offset decay
	recDur         int64   // duration of the current offset decay
	consecOutliers int     // consecutive vision rejections at this point
}

// Tracker fuses IMU and vision into a 6-DOF pose stream on a unified
// timeline. It is safe for concurrent use.
type Tracker struct {
	mu  sync.RWMutex
	cfg Config

	imuClock *clockMap
	visClock *clockMap

	imu imuLog
	vis visionLog

	imuSeqs *seqWindow
	visSeqs *seqWindow

	cps   []checkpoint // genesis checkpoint plus one per accepted vision event
	head  checkpoint   // newest fused state
	snaps []PoseSample // published snapshot ring, sorted by time
}

// NewTracker creates a tracker with the given configuration.
func NewTracker(cfg Config) *Tracker {
	t := &Tracker{
		cfg:      cfg,
		imuClock: newClockMap(cfg.ClockOffsetIMU),
		visClock: newClockMap(cfg.ClockOffsetVision),
		imuSeqs:  newSeqWindow(4096),
		visSeqs:  newSeqWindow(4096),
	}
	genesis := checkpoint{est: newEstimator()}
	t.cps = append(t.cps, genesis)
	t.head = genesis
	return t
}

// PushIMU feeds one inertial sample. Returns the reject reason, or
// RejectNone if the sample was fused.
func (t *Tracker) PushIMU(s IMUSample) RejectReason {
	if !vFinite(s.Accel) || !vFinite(s.Gyro) {
		return RejectInvalid
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.imuSeqs.add(s.Seq) {
		return RejectDuplicate
	}
	ts := t.imuClock.toUnified(s.SensorTime)
	if t.head.est.initialized && ts < t.head.est.time-t.cfg.HistoryWindow {
		return RejectTooLate
	}
	e := imuEntry{Time: ts, Seq: s.Seq, Accel: s.Accel, Gyro: s.Gyro}
	if !t.imu.insert(e) {
		return RejectStaleTimestamp
	}
	if ts >= t.head.est.time {
		// Fast path: extends the head, no replay needed.
		t.head = t.stepIMU(t.head, e)
		t.publish(t.head)
	} else {
		t.recomputeFrom(ts)
	}
	t.prune()
	return RejectNone
}

// PushVision feeds one absolute pose observation. Returns the reject
// reason, or RejectNone if the observation was fused.
func (t *Tracker) PushVision(o VisionObservation) RejectReason {
	if !vFinite(o.Pose.Position) || !qFinite(o.Pose.Orientation) || !isFiniteF(o.Covariance) {
		return RejectInvalid
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.visSeqs.add(o.Seq) {
		return RejectDuplicate
	}
	ts := t.visClock.toUnified(o.SensorTime)
	if t.head.est.initialized && ts < t.head.est.time-t.cfg.HistoryWindow {
		return RejectTooLate
	}
	e := visionEntry{Time: ts, Seq: o.Seq, Pose: o.Pose, Covariance: o.Covariance}
	if !t.vis.insert(e) {
		return RejectStaleTimestamp
	}
	t.recomputeFrom(ts)
	t.prune()
	if !t.vis.has(ts) {
		// The observation was inserted but suppressed by the outlier
		// gate during fusion.
		return RejectOutlier
	}
	return RejectNone
}

// PoseAt returns the pose on the unified timeline at the requested time,
// interpolated between the retained snapshots so output can be aligned
// to any render clock. Times outside the retained range are clamped to
// the nearest available snapshot. The boolean reports whether any pose
// has been published at all.
func (t *Tracker) PoseAt(unifiedTime int64) (PoseSample, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if len(t.snaps) == 0 {
		return PoseSample{}, false
	}
	i := sort.Search(len(t.snaps), func(i int) bool { return t.snaps[i].Time >= unifiedTime })
	if i == 0 {
		return t.snaps[0], true
	}
	if i == len(t.snaps) {
		return t.snaps[len(t.snaps)-1], true
	}
	a, b := t.snaps[i-1], t.snaps[i]
	if b.Time == a.Time {
		return b, true
	}
	f := float64(unifiedTime-a.Time) / float64(b.Time-a.Time)
	return PoseSample{
		Time:    unifiedTime,
		Pose:    poseLerp(a.Pose, b.Pose, f),
		State:   a.State,
		Quality: a.Quality + (b.Quality-a.Quality)*f,
	}, true
}

// Latest returns the most recently published snapshot.
func (t *Tracker) Latest() (PoseSample, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if len(t.snaps) == 0 {
		return PoseSample{}, false
	}
	return t.snaps[len(t.snaps)-1], true
}

// ---- internal pipeline (all called with mu held for writing) ----

// stepIMU propagates cp by one IMU sample. If the state ever becomes
// numerically invalid the sample is discarded and the previous state is
// kept, so a corrupt integration can never poison the output.
func (t *Tracker) stepIMU(cp checkpoint, e imuEntry) checkpoint {
	prev := cp.est
	cp.est.propagate(e, t.cfg.Gravity)
	if !cp.est.finite() {
		cp.est = prev
		cp.est.time = e.Time
	}
	return cp
}

// stepVision applies one logged vision observation to cp. Uncommitted
// observations are gated first; rejected ones stay uncommitted and are
// dropped by the caller. Returns the updated checkpoint and whether the
// observation was accepted.
func (t *Tracker) stepVision(cp checkpoint, o visionEntry, committed bool) (checkpoint, bool) {
	if !cp.haveVision {
		cp.est.initFrom(o)
		cp.haveVision = true
		return cp, true
	}
	if !committed {
		if cp.est.innovation(o) > gateThreshold(t.cfg, o.Covariance) {
			if cp.consecOutliers < t.cfg.MaxConsecutiveOutliers {
				cp.consecOutliers++
				return cp, false
			}
			// Too many consecutive rejections: the scene has genuinely
			// changed. Accept and blend smoothly onto the new frame.
		}
	}
	lost := t.deriveState(cp) == StateLost
	if lost {
		// Smooth recovery: keep the published pose continuous by
		// absorbing the correction into a decaying output offset.
		pub := t.publishPose(cp)
		cp.est.correct(o, t.cfg)
		cp.offPos = vSub(pub.Position, cp.est.pos)
		cp.offRot = qLog(qMul(qConj(cp.est.ori), pub.Orientation))
		cp.recStart = o.Time
		cp.recDur = t.cfg.RecoveryBlend
	} else {
		cp.est.correct(o, t.cfg)
	}
	cp.consecOutliers = 0
	return cp, true
}

// recomputeFrom replays the event logs from the latest checkpoint at or
// before ts, regenerating checkpoints and snapshots up to the head.
func (t *Tracker) recomputeFrom(ts int64) {
	// Find the latest checkpoint at or before ts.
	i := sort.Search(len(t.cps), func(i int) bool { return t.cps[i].est.time > ts }) - 1
	if i < 0 {
		i = 0
	}
	cp := t.cps[i]
	t.cps = t.cps[:i+1]
	t.truncateSnaps(cp.est.time)

	// Committed vision observations are applied as-is; only observations
	// logged since the last recompute (committed == false) are gated.
	ii := t.imu.from(cp.est.time)
	vv := t.vis.from(cp.est.time)
	var rejected, accepted []int64
	i2, j2 := 0, 0
	for i2 < len(ii) || j2 < len(vv) {
		if j2 >= len(vv) || (i2 < len(ii) && ii[i2].Time <= vv[j2].Time) {
			cp = t.stepIMU(cp, ii[i2])
			t.publish(cp)
			i2++
		} else {
			o := vv[j2]
			j2++
			var ok bool
			cp, ok = t.stepVision(cp, o, o.committed)
			if !ok {
				rejected = append(rejected, o.Time)
				continue
			}
			if !o.committed {
				accepted = append(accepted, o.Time)
			}
			t.cps = append(t.cps, cp)
		}
	}
	for _, rt := range rejected {
		t.vis.removeAt(rt)
	}
	for _, at := range accepted {
		t.vis.markCommitted(at)
	}
	t.head = cp
}

// deriveState maps a checkpoint to its tracking state.
func (t *Tracker) deriveState(cp checkpoint) State {
	if !cp.haveVision {
		return StateLost
	}
	dtV := cp.est.time - cp.est.lastVisionAt
	if dtV >= t.cfg.LostTimeout {
		return StateLost
	}
	if cp.recDur > 0 && cp.est.time < cp.recStart+cp.recDur {
		return StateRecovering
	}
	if dtV >= t.cfg.LostTimeout/2 {
		return StateDegraded
	}
	return StateTracking
}

// qualityOf maps a tracking state to a trust score in [0,1].
func qualityOf(s State) float64 {
	switch s {
	case StateTracking:
		return 1
	case StateRecovering:
		return 0.85
	case StateDegraded:
		return 0.6
	case StateLost:
		return 0
	}
	return 0
}

// publishPose computes the output pose of a checkpoint: the fused pose
// plus the decaying recovery offset. The offset decays linearly to zero
// over the recovery blend window, so re-acquisition never jumps.
func (t *Tracker) publishPose(cp checkpoint) Pose {
	rem := 0.0
	if cp.recDur > 0 {
		rem = clampF(1-float64(cp.est.time-cp.recStart)/float64(cp.recDur), 0, 1)
	}
	p := cp.est.pos
	o := cp.est.ori
	if rem > 0 {
		p = vAdd(p, vScale(cp.offPos, rem))
		o = qNormalize(qMul(o, qFromAxisAngle(vScale(cp.offRot, rem))))
	}
	return Pose{Position: p, Orientation: o}
}

// publish appends the snapshot for a checkpoint to the ring.
func (t *Tracker) publish(cp checkpoint) {
	st := t.deriveState(cp)
	t.snaps = append(t.snaps, PoseSample{
		Time:    cp.est.time,
		Pose:    t.publishPose(cp),
		State:   st,
		Quality: qualityOf(st),
	})
	if len(t.snaps) > maxSnapshots {
		t.snaps = append([]PoseSample(nil), t.snaps[len(t.snaps)-maxSnapshots:]...)
	}
}

// truncateSnaps drops snapshots with Time > t (they will be regenerated
// by a replay).
func (t *Tracker) truncateSnaps(t64 int64) {
	i := sort.Search(len(t.snaps), func(i int) bool { return t.snaps[i].Time > t64 })
	t.snaps = t.snaps[:i]
}

// prune drops events and checkpoints older than the history window.
// The newest checkpoint at or before the cut is kept as the replay base.
func (t *Tracker) prune() {
	if !t.head.est.initialized {
		return
	}
	cut := t.head.est.time - t.cfg.HistoryWindow
	t.imu.prune(cut)
	t.vis.prune(cut)
	keep := 0
	for keep < len(t.cps) && t.cps[keep].est.time <= cut {
		keep++
	}
	if keep > 1 {
		t.cps = append([]checkpoint(nil), t.cps[keep-1:]...)
	}
}
