package tracking

import (
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"
)

const (
	imuPeriodNs = int64(5_000_000)     // 200 Hz
	visPeriodNs = int64(33_333_333)    // ~30 Hz
	imuClockOff = int64(1_234_567_000) // sensor clocks differ from unified
	visClockOff = int64(-987_654_000)
	imuLatency  = int64(2e6)  // 2ms
	visLatency  = int64(40e6) // 40ms
)

type streamEvent struct {
	isVision bool
	imu      IMUSample
	vis      VisionObservation
	arrival  int64 // unifiedNow at ingest
}

// genStream builds a deterministic sensor stream. Ground truth: the
// device moves along +X at 0.5 m/s, identity orientation, duration
// seconds long.
func genStream(seed int64, duration time.Duration) []streamEvent {
	rng := rand.New(rand.NewSource(seed))
	noise := func(std float64) float64 { return rng.NormFloat64() * std }

	var events []streamEvent
	imuPeriod := imuPeriodNs
	visPeriod := visPeriodNs
	total := duration.Nanoseconds()

	for ts := int64(0); ts <= total; ts += imuPeriod {
		events = append(events, streamEvent{
			imu: IMUSample{
				Timestamp: ts + imuClockOff,
				Accel: Vec3{
					X: noise(0.1),
					Y: noise(0.1),
					Z: 9.81 + noise(0.1),
				},
				Gyro: Vec3{X: noise(0.01), Y: noise(0.01), Z: noise(0.01)},
			},
			arrival: ts + imuLatency,
		})
	}
	for ts := int64(0); ts <= total; ts += visPeriod {
		sec := float64(ts) * 1e-9
		events = append(events, streamEvent{
			isVision: true,
			vis: VisionObservation{
				Timestamp: ts + visClockOff,
				Pose: Pose{
					Position:    Vec3{X: 0.5*sec + noise(0.01), Y: noise(0.01), Z: noise(0.01)},
					Orientation: Quat{W: 1, X: noise(0.002), Y: noise(0.002), Z: noise(0.002)},
				},
			},
			arrival: ts + visLatency,
		})
	}
	// Sort by arrival (this is the order a real runtime sees them).
	for i := 1; i < len(events); i++ {
		for j := i; j > 0 && events[j].arrival < events[j-1].arrival; j-- {
			events[j], events[j-1] = events[j-1], events[j]
		}
	}
	return events
}

func runStream(tr *Tracker, events []streamEvent) {
	for _, e := range events {
		if e.isVision {
			tr.IngestVision(e.vis, e.arrival)
		} else {
			tr.IngestIMU(e.imu, e.arrival)
		}
	}
}

func queryTimes(start, end, step int64) []int64 {
	var out []int64
	for ts := start; ts <= end; ts += step {
		out = append(out, ts)
	}
	return out
}

func snapshotsAt(tr *Tracker, times []int64) []PoseSnapshot {
	out := make([]PoseSnapshot, len(times))
	for i, ts := range times {
		out[i] = tr.PoseAt(ts)
	}
	return out
}

func truthPos(unifiedTs int64) Vec3 {
	return Vec3{X: 0.5 * float64(unifiedTs) * 1e-9}
}

func TestDeterministicReproduction(t *testing.T) {
	events := genStream(42, 5*time.Second)
	queries := queryTimes(int64(500e6), int64(4.5e9), int64(7e6))

	a := NewTracker(Config{})
	runStream(a, events)
	sa := snapshotsAt(a, queries)

	b := NewTracker(Config{})
	runStream(b, events)
	sb := snapshotsAt(b, queries)

	if len(sa) != len(sb) {
		t.Fatalf("snapshot count mismatch")
	}
	for i := range sa {
		if sa[i] != sb[i] {
			t.Fatalf("snapshot %d differs between identical runs:\n%+v\n%+v", i, sa[i], sb[i])
		}
	}
}

func TestTracksGroundTruth(t *testing.T) {
	events := genStream(7, 5*time.Second)
	tr := NewTracker(Config{})
	runStream(tr, events)
	for _, ts := range queryTimes(int64(2e9), int64(4.9e9), int64(100e6)) {
		s := tr.PoseAt(ts)
		want := truthPos(ts)
		err := vNorm(vSub(s.Pose.Position, want))
		if err > 0.15 {
			t.Fatalf("t=%d position error %v m (got %+v want %+v)", ts, err, s.Pose.Position, want)
		}
		if s.State != StateTracking && s.State != StateRecovering {
			t.Fatalf("t=%d unexpected state %v", ts, s.State)
		}
	}
}

func TestOutOfOrderVision(t *testing.T) {
	base := genStream(11, 5*time.Second)

	// Delay every 3rd vision observation by 60ms (within the 100ms
	// reprocessing window) so it arrives out of order.
	delayed := make([]streamEvent, len(base))
	copy(delayed, base)
	for i := range delayed {
		if delayed[i].isVision && i%3 == 0 {
			delayed[i].arrival += 60e6
		}
	}
	for i := 1; i < len(delayed); i++ {
		for j := i; j > 0 && delayed[j].arrival < delayed[j-1].arrival; j-- {
			delayed[j], delayed[j-1] = delayed[j-1], delayed[j]
		}
	}

	queries := queryTimes(int64(3e9), int64(4.9e9), int64(50e6))
	trA := NewTracker(Config{})
	runStream(trA, base)
	trB := NewTracker(Config{})
	runStream(trB, delayed)

	for _, ts := range queries {
		a, b := trA.PoseAt(ts), trB.PoseAt(ts)
		d := vNorm(vSub(a.Pose.Position, b.Pose.Position))
		if d > 0.05 {
			t.Fatalf("t=%d out-of-order result diverges by %v m", ts, d)
		}
		want := truthPos(ts)
		if err := vNorm(vSub(b.Pose.Position, want)); err > 0.15 {
			t.Fatalf("t=%d out-of-order run error %v m vs truth", ts, err)
		}
	}
}

func TestTooLateVisionDropped(t *testing.T) {
	tr := NewTracker(Config{})
	events := genStream(13, 3*time.Second)
	runStream(tr, events)
	// A vision observation 5s in the past must be dropped as too-late.
	late := VisionObservation{
		Timestamp: visClockOff + 1000, // unified ~1us, far behind frontier
		Pose:      Pose{Position: Vec3{X: 99}, Orientation: Quat{W: 1}},
	}
	if r := tr.IngestVision(late, 3_000_000_000); r != RejectTooLate {
		t.Fatalf("expected too-late reject, got %v", r)
	}
}

func TestDuplicateRejected(t *testing.T) {
	events := genStream(17, 3*time.Second)
	queries := queryTimes(int64(2e9), int64(2.9e9), int64(33e6))

	single := NewTracker(Config{})
	runStream(single, events)
	want := snapshotsAt(single, queries)

	dup := NewTracker(Config{})
	dupCount := 0
	for _, e := range events {
		if e.isVision {
			dup.IngestVision(e.vis, e.arrival)
			if r := dup.IngestVision(e.vis, e.arrival); r == RejectDuplicate {
				dupCount++
			}
		} else {
			dup.IngestIMU(e.imu, e.arrival)
			dup.IngestIMU(e.imu, e.arrival) // duplicate IMU must not double-integrate
		}
	}
	if dupCount == 0 {
		t.Fatalf("no duplicates were detected")
	}
	got := snapshotsAt(dup, queries)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("duplicate ingestion changed estimate at %d:\n%+v\n%+v", i, got[i], want[i])
		}
	}
}

func TestOutlierSuppression(t *testing.T) {
	tr := NewTracker(Config{})
	events := genStream(19, 3*time.Second)
	runStream(tr, events)

	// A wildly wrong observation must be gated with a distinct reason.
	bad := VisionObservation{
		Timestamp: visClockOff + 3_100_000_000,
		Pose:      Pose{Position: Vec3{X: 500, Y: 0, Z: 0}, Orientation: Quat{W: 1}},
	}
	if r := tr.IngestVision(bad, 3_100_000_000+visLatency); r != RejectOutlierPosition {
		t.Fatalf("expected outlier-position reject, got %v", r)
	}
	s := tr.PoseAt(3_100_000_000)
	if err := vNorm(vSub(s.Pose.Position, truthPos(3_100_000_000))); err > 0.5 {
		t.Fatalf("estimate polluted by outlier: error %v m", err)
	}
	found := false
	for _, rj := range tr.Rejects() {
		if rj.Reason == RejectOutlierPosition {
			found = true
		}
	}
	if !found {
		t.Fatalf("reject log missing outlier entry")
	}
}

func TestConsecutiveOutliersDoNotDiverge(t *testing.T) {
	tr := NewTracker(Config{})
	runStream(tr, genStream(23, 2*time.Second))
	// Feed a sustained stream of bad observations.
	for i := 0; i < 30; i++ {
		ts := int64(2_100_000_000) + int64(i)*33_000_000
		bad := VisionObservation{
			Timestamp: visClockOff + ts,
			Pose:      Pose{Position: Vec3{X: 1000 + float64(i)}, Orientation: Quat{W: 1}},
		}
		tr.IngestVision(bad, ts+visLatency)
		for j := 0; j < 7; j++ {
			its := ts + int64(j)*5_000_000
			tr.IngestIMU(IMUSample{Timestamp: imuClockOff + its,
				Accel: Vec3{Z: 9.81}}, its+imuLatency)
		}
	}
	s := tr.PoseAt(3_000_000_000)
	if !isFiniteVec(s.Pose.Position) || !isFiniteQuat(s.Pose.Orientation) {
		t.Fatalf("estimate became non-finite under outlier stream")
	}
	if math.IsNaN(s.CovarianceTrace) || s.CovarianceTrace > 1000 {
		t.Fatalf("covariance diverged: %v", s.CovarianceTrace)
	}
}

func TestTrackingLossAndRecovery(t *testing.T) {
	tr := NewTracker(Config{})
	events := genStream(29, 2*time.Second)
	runStream(tr, events)

	// 2s of IMU-only (vision lost).
	lostStart := int64(2_000_000_000)
	for ts := lostStart; ts <= lostStart+2_000_000_000; ts += 5_000_000 {
		tr.IngestIMU(IMUSample{Timestamp: imuClockOff + ts,
			Accel: Vec3{Z: 9.81}}, ts+imuLatency)
	}
	if st := tr.State(); st != StateLost {
		t.Fatalf("expected lost state, got %v", st)
	}
	lostSnap := tr.PoseAt(lostStart + 2_000_000_000)
	if lostSnap.State != StateLost {
		t.Fatalf("snapshot not marked untrusted: %v", lostSnap.State)
	}

	// Re-acquire: vision returns at truth.
	reacq := lostStart + 2_033_000_000
	before := tr.PoseAt(reacq)
	sec := float64(reacq) * 1e-9
	r := tr.IngestVision(VisionObservation{
		Timestamp: visClockOff + reacq,
		Pose:      Pose{Position: Vec3{X: 0.5 * sec}, Orientation: Quat{W: 1}},
	}, reacq+visLatency)
	if r != RejectNone {
		t.Fatalf("re-acquisition observation rejected: %v", r)
	}
	after := tr.PoseAt(reacq)
	jump := vNorm(vSub(after.Pose.Position, before.Pose.Position))
	if jump > 0.05 {
		t.Fatalf("output jumped %v m on re-acquisition", jump)
	}
	if st := tr.State(); st != StateRecovering && st != StateTracking {
		t.Fatalf("expected recovering/tracking after re-acquire, got %v", st)
	}

	// Continue feeding vision; the blend must converge back to truth.
	for ts := reacq + 33_000_000; ts <= reacq+1_000_000_000; ts += 33_000_000 {
		sec := float64(ts) * 1e-9
		tr.IngestVision(VisionObservation{
			Timestamp: visClockOff + ts,
			Pose:      Pose{Position: Vec3{X: 0.5 * sec}, Orientation: Quat{W: 1}},
		}, ts+visLatency)
		for j := 0; j < 7; j++ {
			its := ts + int64(j)*5_000_000
			tr.IngestIMU(IMUSample{Timestamp: imuClockOff + its,
				Accel: Vec3{Z: 9.81}}, its+imuLatency)
		}
	}
	end := reacq + 1_000_000_000
	final := tr.PoseAt(end)
	if err := vNorm(vSub(final.Pose.Position, truthPos(end))); err > 0.2 {
		t.Fatalf("did not converge after recovery, error %v m", err)
	}
	if st := tr.State(); st != StateTracking {
		t.Fatalf("expected tracking after recovery, got %v", st)
	}
}

func TestLongRunStability(t *testing.T) {
	// 120 simulated seconds with a constant gyro bias: error must stay
	// bounded and the estimate must remain finite and trusted.
	tr := NewTracker(Config{})
	events := genStream(31, 120*time.Second)
	for i := range events {
		if !events[i].isVision {
			events[i].imu.Gyro.Z += 0.002 // constant bias
		}
	}
	runStream(tr, events)
	end := int64(119_000_000_000)
	s := tr.PoseAt(end)
	if !isFiniteVec(s.Pose.Position) || !isFiniteQuat(s.Pose.Orientation) {
		t.Fatalf("non-finite estimate after long run")
	}
	if err := vNorm(vSub(s.Pose.Position, truthPos(end))); err > 1.0 {
		t.Fatalf("unbounded drift after 120s: %v m", err)
	}
	if math.IsNaN(s.CovarianceTrace) || math.IsInf(s.CovarianceTrace, 0) {
		t.Fatalf("invalid covariance after long run")
	}
	if s.State != StateTracking {
		t.Fatalf("unexpected state after long run: %v", s.State)
	}
}

func TestNonFiniteRejected(t *testing.T) {
	tr := NewTracker(Config{})
	runStream(tr, genStream(37, time.Second))
	nan := math.NaN()
	if r := tr.IngestVision(VisionObservation{
		Timestamp: visClockOff + 1_100_000_000,
		Pose:      Pose{Position: Vec3{X: nan}, Orientation: Quat{W: 1}},
	}, 1_100_000_000); r != RejectNonFinite {
		t.Fatalf("expected non-finite reject, got %v", r)
	}
	tr.IngestIMU(IMUSample{Timestamp: imuClockOff + 1_100_000_000,
		Accel: Vec3{X: math.Inf(1)}}, 1_100_000_000)
	found := 0
	for _, rj := range tr.Rejects() {
		if rj.Reason == RejectNonFinite {
			found++
		}
	}
	if found < 2 {
		t.Fatalf("expected non-finite rejects in log, got %d", found)
	}
}

func TestConcurrentQueries(t *testing.T) {
	tr := NewTracker(Config{})
	events := genStream(41, 10*time.Second)

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			i := int64(0)
			for {
				select {
				case <-stop:
					return
				default:
				}
				ts := int64(100e6) + (i%90)*100_000_000
				s := tr.PoseAt(ts)
				if !isFiniteVec(s.Pose.Position) || !isFiniteQuat(s.Pose.Orientation) ||
					!isFiniteVec(s.Velocity) {
					t.Errorf("inconsistent snapshot under concurrency: %+v", s)
					return
				}
				// A self-consistent snapshot always has a unit
				// quaternion.
				n := s.Pose.Orientation.W*s.Pose.Orientation.W +
					s.Pose.Orientation.X*s.Pose.Orientation.X +
					s.Pose.Orientation.Y*s.Pose.Orientation.Y +
					s.Pose.Orientation.Z*s.Pose.Orientation.Z
				if math.Abs(n-1) > 1e-6 {
					t.Errorf("non-unit quaternion in snapshot: %v", n)
					return
				}
				i++
			}
		}()
	}
	runStream(tr, events)
	close(stop)
	wg.Wait()
}

func TestInterpolationBetweenSamples(t *testing.T) {
	tr := NewTracker(Config{})
	runStream(tr, genStream(43, 3*time.Second))
	// Query exactly between two IMU steps; the result must lie between
	// the neighboring snapshots.
	a := tr.PoseAt(2_000_000_000)
	b := tr.PoseAt(2_005_000_000)
	mid := tr.PoseAt(2_002_500_000)
	lo := math.Min(a.Pose.Position.X, b.Pose.Position.X) - 0.05
	hi := math.Max(a.Pose.Position.X, b.Pose.Position.X) + 0.05
	if mid.Pose.Position.X < lo || mid.Pose.Position.X > hi {
		t.Fatalf("interpolation out of bounds: %v not in [%v,%v]",
			mid.Pose.Position.X, lo, hi)
	}
	if mid.Timestamp != 2_002_500_000 {
		t.Fatalf("snapshot not aligned to requested time: %d", mid.Timestamp)
	}
}
