// Package tracking implements a 6-DOF pose tracking runtime that fuses
// inertial measurements with absolute vision observations onto a unified
// timeline. It handles out-of-order, late and duplicate observations,
// suppresses outliers, recovers smoothly after tracking loss, and serves
// consistent pose snapshots to concurrent readers.
package tracking

// Vector3 is a 3D vector in meters or radians depending on context.
type Vector3 struct {
	X, Y, Z float64
}

// Quaternion is a unit rotation quaternion (w is the scalar part).
type Quaternion struct {
	W, X, Y, Z float64
}

// Pose is a rigid-body pose: position in meters and orientation.
type Pose struct {
	Position    Vector3
	Orientation Quaternion
}

// IMUSample is one inertial measurement. SensorTime is in nanoseconds on
// the IMU's own clock; it is mapped to the unified timeline by the
// tracker's clock model. Seq is a per-stream monotonically increasing
// sequence number used for duplicate detection.
type IMUSample struct {
	Seq        uint64
	SensorTime int64
	Accel      Vector3 // m/s^2, specific force in body frame
	Gyro       Vector3 // rad/s, angular velocity in body frame
}

// VisionObservation is one absolute pose observation from the vision
// pipeline. SensorTime is in nanoseconds on the vision clock. Covariance
// is a scalar uncertainty proxy (larger means less trusted).
type VisionObservation struct {
	Seq        uint64
	SensorTime int64
	Pose       Pose
	Covariance float64
}

// State describes the tracker's confidence in its output.
type State int

const (
	// StateTracking means vision and IMU are fused normally.
	StateTracking State = iota
	// StateDegraded means vision is intermittent; output is still usable.
	StateDegraded
	// StateLost means no absolute observation for too long; output is
	// inertial-only dead reckoning and must be treated as untrusted.
	StateLost
	// StateRecovering means an absolute observation was re-acquired and
	// the output is being blended back onto the absolute frame.
	StateRecovering
)

func (s State) String() string {
	switch s {
	case StateTracking:
		return "tracking"
	case StateDegraded:
		return "degraded"
	case StateLost:
		return "lost"
	case StateRecovering:
		return "recovering"
	}
	return "unknown"
}

// RejectReason explains why an observation was not fused.
type RejectReason int

const (
	RejectNone RejectReason = iota
	// RejectDuplicate: same stream sequence number already consumed.
	RejectDuplicate
	// RejectTooLate: timestamp is older than the retained history window.
	RejectTooLate
	// RejectOutlier: observation is inconsistent with the prediction.
	RejectOutlier
	// RejectInvalid: observation contains NaN/Inf or non-normalized data.
	RejectInvalid
	// RejectStaleTimestamp: timestamp does not advance the stream clock.
	RejectStaleTimestamp
)

func (r RejectReason) String() string {
	switch r {
	case RejectNone:
		return "none"
	case RejectDuplicate:
		return "duplicate"
	case RejectTooLate:
		return "too-late"
	case RejectOutlier:
		return "outlier"
	case RejectInvalid:
		return "invalid"
	case RejectStaleTimestamp:
		return "stale-timestamp"
	}
	return "unknown"
}

// PoseSample is an immutable, self-consistent pose snapshot on the
// unified timeline. Readers always receive a whole PoseSample; fields of
// different samples are never mixed.
type PoseSample struct {
	Time    int64 // unified timeline, nanoseconds
	Pose    Pose
	State   State
	Quality float64 // 0..1, 1 means fully trusted
}

// Config holds all tunables. Durations are in nanoseconds.
type Config struct {
	// ClockOffsetIMU / ClockOffsetVision map sensor clocks onto the
	// unified timeline: unified = sensor + offset.
	ClockOffsetIMU    int64
	ClockOffsetVision int64

	// HistoryWindow is how far back the tracker can accept late
	// observations and replay. Older observations are dropped.
	HistoryWindow int64

	// LostTimeout: no accepted vision observation for this long puts the
	// tracker into StateLost.
	LostTimeout int64

	// RecoveryBlend: duration over which the output is blended from the
	// dead-reckoned pose to the re-acquired absolute pose.
	RecoveryBlend int64

	// OutlierGate: maximum allowed normalized innovation (position in
	// meters + rotation in radians combined) before an observation is
	// suppressed as an outlier.
	OutlierGate float64

	// MaxConsecutiveOutliers: after this many consecutive suppressed
	// observations the tracker treats the stream as re-acquired at a new
	// absolute frame and accepts the next observation, preventing
	// divergence when the scene genuinely changed.
	MaxConsecutiveOutliers int

	// GyroBiasAlpha: learning rate for online gyro bias estimation while
	// vision is available. Bounds long-run drift.
	GyroBiasAlpha float64

	// Gravity is the gravity vector in the world frame.
	Gravity Vector3
}

// DefaultConfig returns a configuration suitable for typical
// head-mounted or mobile devices.
func DefaultConfig() Config {
	return Config{
		ClockOffsetIMU:         0,
		ClockOffsetVision:      0,
		HistoryWindow:          2_000_000_000, // 2s
		LostTimeout:            500_000_000,   // 500ms
		RecoveryBlend:          300_000_000,   // 300ms
		OutlierGate:            1.5,
		MaxConsecutiveOutliers: 5,
		GyroBiasAlpha:          0.001,
		Gravity:                Vector3{X: 0, Y: 0, Z: -9.81},
	}
}
