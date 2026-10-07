package tracking

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

// Deterministic simulation harness. Ground truth: the device moves along
// +x at 0.5 m/s with identity orientation. The IMU reports specific
// force (0,0,9.81) and a constant gyro bias; vision reports the true
// pose plus seeded noise.

const (
	imuPeriodNs = 5_000_000  // 200 Hz
	visPeriodNs = 33_333_333 // ~30 Hz
	speed       = 0.5
	gyroBiasZ   = 0.02
)

func truePos(tNs int64) Vector3 {
	return Vector3{X: speed * float64(tNs) * 1e-9}
}

func imuAt(seq uint64, tNs int64) IMUSample {
	return IMUSample{
		Seq:        seq,
		SensorTime: tNs,
		Accel:      Vector3{Z: 9.81},
		Gyro:       Vector3{Z: gyroBiasZ},
	}
}

func visionAt(seq uint64, tNs int64, r *rand.Rand) VisionObservation {
	p := truePos(tNs)
	if r != nil {
		p.X += (r.Float64() - 0.5) * 0.02
		p.Y += (r.Float64() - 0.5) * 0.02
		p.Z += (r.Float64() - 0.5) * 0.02
	}
	return VisionObservation{
		Seq:        seq,
		SensorTime: tNs,
		Pose:       Pose{Position: p, Orientation: Quaternion{W: 1}},
		Covariance: 0.01,
	}
}

func posErr(s PoseSample, tNs int64) float64 {
	return vNorm(vSub(s.Pose.Position, truePos(tNs)))
}

// feed runs a scripted scenario: events are generated on the unified
// timeline and delivered via deliver, which may reorder them.
func feed(tr *Tracker, durationNs int64, r *rand.Rand, visionGaps [][2]int64, deliver func(ev event)) {
	type ev struct {
		t   int64
		imu *IMUSample
		vis *VisionObservation
	}
	var imuEvents, visEvents []ev
	var iseq, vseq uint64
	for t := int64(0); t <= durationNs; t += imuPeriodNs {
		s := imuAt(iseq, t)
		iseq++
		imuEvents = append(imuEvents, ev{t: t, imu: &s})
	}
nextVis:
	for t := int64(visPeriodNs); t <= durationNs; t += visPeriodNs {
		for _, g := range visionGaps {
			if t >= g[0] && t < g[1] {
				continue nextVis
			}
		}
		o := visionAt(vseq, t, r)
		vseq++
		visEvents = append(visEvents, ev{t: t, vis: &o})
	}
	// Merge the two streams in unified-time order.
	var events []ev
	i, j := 0, 0
	for i < len(imuEvents) || j < len(visEvents) {
		if j >= len(visEvents) || (i < len(imuEvents) && imuEvents[i].t <= visEvents[j].t) {
			events = append(events, imuEvents[i])
			i++
		} else {
			events = append(events, visEvents[j])
			j++
		}
	}
	for _, e := range events {
		if e.imu != nil {
			deliver(event{imu: e.imu})
		} else {
			deliver(event{vis: e.vis})
		}
	}
}

type event struct {
	imu *IMUSample
	vis *VisionObservation
}

func TestFusionAlignsToRequestedTime(t *testing.T) {
	tr := NewTracker(DefaultConfig())
	r := rand.New(rand.NewSource(1))
	feed(tr, 2_000_000_000, r, nil, func(e event) {
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			tr.PushVision(*e.vis)
		}
	})
	// Query at a time that falls between two IMU samples.
	q := int64(1_502_500_000)
	s, ok := tr.PoseAt(q)
	if !ok {
		t.Fatal("no pose available")
	}
	if s.Time != q {
		t.Fatalf("expected aligned time %d, got %d", q, s.Time)
	}
	if s.State != StateTracking {
		t.Fatalf("expected tracking, got %v", s.State)
	}
	if err := posErr(s, q); err > 0.05 {
		t.Fatalf("position error too large: %v", err)
	}
}

func TestLateAndOutOfOrder(t *testing.T) {
	cfg := DefaultConfig()
	tr := NewTracker(cfg)
	r := rand.New(rand.NewSource(2))
	// Deliver events mostly in order, but hold back every 40th IMU
	// sample and every 12th vision observation and deliver them late.
	var held []event
	count := 0
	deliver := func(e event) {
		count++
		if count%40 == 0 {
			held = append(held, e)
			return
		}
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			tr.PushVision(*e.vis)
		}
		// Flush held events that are still inside the history window.
		for _, h := range held {
			if h.imu != nil {
				tr.PushIMU(*h.imu)
			} else {
				tr.PushVision(*h.vis)
			}
		}
		held = nil
	}
	feed(tr, 2_000_000_000, r, nil, deliver)
	for _, h := range held {
		if h.imu != nil {
			tr.PushIMU(*h.imu)
		} else {
			tr.PushVision(*h.vis)
		}
	}
	s, ok := tr.Latest()
	if !ok {
		t.Fatal("no pose")
	}
	if err := posErr(s, s.Time); err > 0.1 {
		t.Fatalf("out-of-order run diverged: err=%v", err)
	}

	// An observation older than the history window must be dropped.
	late := visionAt(90000, s.Time-cfg.HistoryWindow-1_000_000, nil)
	if got := tr.PushVision(late); got != RejectTooLate {
		t.Fatalf("expected too-late, got %v", got)
	}
	lateIMU := imuAt(90001, s.Time-cfg.HistoryWindow-1_000_000)
	if got := tr.PushIMU(lateIMU); got != RejectTooLate {
		t.Fatalf("expected too-late for IMU, got %v", got)
	}
}

func TestDuplicateAndStaleRejected(t *testing.T) {
	tr := NewTracker(DefaultConfig())
	r := rand.New(rand.NewSource(3))
	feed(tr, 500_000_000, r, nil, func(e event) {
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			tr.PushVision(*e.vis)
		}
	})
	// Exact duplicate (same sequence number).
	dup := visionAt(5, 6*visPeriodNs, nil)
	if got := tr.PushVision(dup); got != RejectDuplicate {
		t.Fatalf("expected duplicate, got %v", got)
	}
	dupIMU := imuAt(10, 10*imuPeriodNs)
	if got := tr.PushIMU(dupIMU); got != RejectDuplicate {
		t.Fatalf("expected duplicate IMU, got %v", got)
	}
	// New sequence number but an already-consumed timestamp.
	stale := visionAt(77777, 6*visPeriodNs, nil)
	if got := tr.PushVision(stale); got != RejectStaleTimestamp {
		t.Fatalf("expected stale-timestamp, got %v", got)
	}
}

func TestOutlierSuppression(t *testing.T) {
	cfg := DefaultConfig()
	tr := NewTracker(cfg)
	r := rand.New(rand.NewSource(4))
	feed(tr, 1_000_000_000, r, nil, func(e event) {
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			tr.PushVision(*e.vis)
		}
	})
	base := int64(1_000_000_000)
	// Inject clearly-wrong observations; all must be suppressed with a
	// distinguishable reason and must not corrupt the estimate.
	for i := 0; i < cfg.MaxConsecutiveOutliers; i++ {
		ts := base + int64(i+1)*imuPeriodNs
		tr.PushIMU(imuAt(10000+uint64(i), ts))
		bad := visionAt(20000+uint64(i), ts, nil)
		bad.Pose.Position.X += 10
		if got := tr.PushVision(bad); got != RejectOutlier {
			t.Fatalf("outlier %d: expected reject, got %v", i, got)
		}
	}
	s, _ := tr.Latest()
	if err := posErr(s, s.Time); err > 0.2 {
		t.Fatalf("estimate polluted by outliers: err=%v", err)
	}
	// NaN input is rejected as invalid.
	nan := visionAt(30000, base+100*imuPeriodNs, nil)
	nan.Pose.Position.X = math.NaN()
	if got := tr.PushVision(nan); got != RejectInvalid {
		t.Fatalf("expected invalid, got %v", got)
	}
}

func TestLossAndSmoothRecovery(t *testing.T) {
	cfg := DefaultConfig()
	tr := NewTracker(cfg)
	r := rand.New(rand.NewSource(5))
	// Vision blackout between 1s and 2s.
	gap := [][2]int64{{1_000_000_000, 2_000_000_000}}
	feed(tr, 3_000_000_000, r, gap, func(e event) {
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			tr.PushVision(*e.vis)
		}
	})
	// During the blackout the output must be marked untrusted.
	mid, ok := tr.PoseAt(1_800_000_000)
	if !ok {
		t.Fatal("no pose during blackout")
	}
	if mid.State != StateLost || mid.Quality != 0 {
		t.Fatalf("expected lost/untrusted, got %v q=%v", mid.State, mid.Quality)
	}
	// After reacquisition the tracker recovers and converges again.
	end, _ := tr.Latest()
	if end.State != StateTracking {
		t.Fatalf("expected tracking after recovery, got %v", end.State)
	}
	if err := posErr(end, end.Time); err > 0.1 {
		t.Fatalf("did not converge after recovery: err=%v", err)
	}
	// Smoothness: no jump at reacquisition. Adjacent snapshots around
	// the first vision observation after the gap must be close.
	first := int64(2_000_000_000 + visPeriodNs)
	prev, ok1 := tr.PoseAt(first - 1)
	next, ok2 := tr.PoseAt(first + imuPeriodNs)
	if !ok1 || !ok2 {
		t.Fatal("missing snapshots around recovery")
	}
	jump := vNorm(vSub(next.Pose.Position, prev.Pose.Position))
	if jump > 0.05 {
		t.Fatalf("pose jumped at recovery: %v m", jump)
	}
	// The recovering state must be observable.
	rec, _ := tr.PoseAt(first + cfg.RecoveryBlend/2)
	if rec.State != StateRecovering {
		t.Fatalf("expected recovering, got %v", rec.State)
	}
}

func TestLongRunDriftBounded(t *testing.T) {
	tr := NewTracker(DefaultConfig())
	r := rand.New(rand.NewSource(6))
	feed(tr, 60_000_000_000, r, nil, func(e event) {
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			tr.PushVision(*e.vis)
		}
	})
	end, _ := tr.Latest()
	if !vFinite(end.Pose.Position) || !qFinite(end.Pose.Orientation) {
		t.Fatal("non-finite output after long run")
	}
	if err := posErr(end, end.Time); err > 0.05 {
		t.Fatalf("long-run drift not bounded: err=%v", err)
	}
}

func TestConcurrentQueries(t *testing.T) {
	tr := NewTracker(DefaultConfig())
	r := rand.New(rand.NewSource(7))
	var wg sync.WaitGroup
	stop := make(chan struct{})
	// Readers: hammer PoseAt from several goroutines.
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rr := rand.New(rand.NewSource(seed))
			for {
				select {
				case <-stop:
					return
				default:
				}
				q := rr.Int63n(2_000_000_000)
				if s, ok := tr.PoseAt(q); ok {
					if !vFinite(s.Pose.Position) || !qFinite(s.Pose.Orientation) {
						t.Error("non-finite snapshot")
						return
					}
				}
			}
		}(100 + int64(i))
	}
	feed(tr, 2_000_000_000, r, nil, func(e event) {
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			tr.PushVision(*e.vis)
		}
	})
	close(stop)
	wg.Wait()
}

func TestDeterminism(t *testing.T) {
	run := func() []PoseSample {
		tr := NewTracker(DefaultConfig())
		r := rand.New(rand.NewSource(42))
		var held []event
		count := 0
		feed(tr, 3_000_000_000, r, [][2]int64{{1_000_000_000, 1_500_000_000}}, func(e event) {
			count++
			// Deterministic reordering: hold every 25th event briefly.
			if count%25 == 0 {
				held = append(held, e)
				return
			}
			if e.imu != nil {
				tr.PushIMU(*e.imu)
			} else {
				tr.PushVision(*e.vis)
			}
			if len(held) > 2 {
				h := held[0]
				held = held[1:]
				if h.imu != nil {
					tr.PushIMU(*h.imu)
				} else {
					tr.PushVision(*h.vis)
				}
			}
		})
		var out []PoseSample
		for q := int64(0); q <= 3_000_000_000; q += 7_654_321 {
			if s, ok := tr.PoseAt(q); ok {
				out = append(out, s)
			}
		}
		return out
	}
	a := run()
	b := run()
	if len(a) != len(b) || len(a) == 0 {
		t.Fatalf("snapshot count mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("snapshot %d differs:\n%+v\n%+v", i, a[i], b[i])
		}
	}
}

func TestClockOffsets(t *testing.T) {
	cfg := DefaultConfig()
	// The vision clock runs 40ms ahead of the unified clock; the
	// configured offset maps it back.
	cfg.ClockOffsetVision = -40_000_000
	tr := NewTracker(cfg)
	r := rand.New(rand.NewSource(8))
	feed(tr, 2_000_000_000, r, nil, func(e event) {
		if e.imu != nil {
			tr.PushIMU(*e.imu)
		} else {
			o := *e.vis
			o.SensorTime += 40_000_000 // vision clock is ahead
			if got := tr.PushVision(o); got != RejectNone {
				t.Fatalf("vision rejected: %v", got)
			}
		}
	})
	s, ok := tr.Latest()
	if !ok {
		t.Fatal("no pose")
	}
	if err := posErr(s, s.Time); err > 0.05 {
		t.Fatalf("clock offset not compensated: err=%v", err)
	}
}
