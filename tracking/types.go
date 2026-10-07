// Package tracking provides a 6-DoF pose tracking runtime that fuses
// inertial measurements with absolute vision observations onto a unified
// timeline. It is designed for head-mounted and mobile devices where
// sensors sample at different rates on different clocks, observations
// arrive out of order, and tracking loss must be recovered smoothly.
package tracking

import "time"

// Vec3 is a 3D vector in a right-handed coordinate frame.
type Vec3 struct {
	X, Y, Z float64
}

// Quat is a unit quaternion (w is the scalar part) representing a
// rotation from the body frame to the world frame.
type Quat struct {
	W, X, Y, Z float64
}

// Pose is a rigid transform: position and orientation in world frame.
type Pose struct {
	Position    Vec3
	Orientation Quat
}

// ClockID identifies the clock domain a timestamp was sampled on.
type ClockID uint8

const (
	// ClockIMU is the (typically monotonic, device-local) IMU clock.
	ClockIMU ClockID = iota
	// ClockVision is the camera/vision pipeline clock.
	ClockVision
)

// IMUSample is one inertial measurement.
type IMUSample struct {
	// Timestamp is in nanoseconds on ClockIMU.
	Timestamp int64
	// Accel is specific force in m/s^2 in the body frame.
	Accel Vec3
	// Gyro is angular velocity in rad/s in the body frame.
	Gyro Vec3
}

// VisionObservation is an absolute 6-DoF pose measurement from the
// vision pipeline.
type VisionObservation struct {
	// Timestamp is in nanoseconds on ClockVision.
	Timestamp int64
	Pose      Pose
	// PositionStd and OrientationStd are 1-sigma measurement noise
	// estimates (meters, radians). Zero values fall back to config
	// defaults.
	PositionStd    float64
	OrientationStd float64
}

// TrackingState describes the health of the estimator.
type TrackingState uint8

const (
	// StateInitializing means no absolute observation has been fused yet;
	// the pose is relative to the start pose.
	StateInitializing TrackingState = iota
	// StateTracking means absolute observations are being fused normally.
	StateTracking
	// StateLost means vision observations have been missing longer than
	// the configured timeout; the pose is inertial-only and untrusted.
	StateLost
	// StateRecovering means an absolute observation was re-acquired and
	// the output is being smoothly blended back to the fused estimate.
	StateRecovering
)

func (s TrackingState) String() string {
	switch s {
	case StateInitializing:
		return "initializing"
	case StateTracking:
		return "tracking"
	case StateLost:
		return "lost"
	case StateRecovering:
		return "recovering"
	}
	return "unknown"
}

// RejectReason explains why an observation was not fused.
type RejectReason uint8

const (
	RejectNone RejectReason = iota
	// RejectDuplicate: an observation with the same timestamp was
	// already ingested.
	RejectDuplicate
	// RejectTooLate: the observation arrived after the estimator had
	// advanced beyond the reprocessing horizon.
	RejectTooLate
	// RejectOutlierPosition: position innovation exceeded the gate.
	RejectOutlierPosition
	// RejectOutlierOrientation: orientation innovation exceeded the gate.
	RejectOutlierOrientation
	// RejectNonFinite: the observation contained NaN or Inf values.
	RejectNonFinite
)

func (r RejectReason) String() string {
	switch r {
	case RejectNone:
		return "accepted"
	case RejectDuplicate:
		return "duplicate"
	case RejectTooLate:
		return "too-late"
	case RejectOutlierPosition:
		return "outlier-position"
	case RejectOutlierOrientation:
		return "outlier-orientation"
	case RejectNonFinite:
		return "non-finite"
	}
	return "unknown"
}

// PoseSnapshot is an immutable, self-consistent pose query result.
type PoseSnapshot struct {
	// Timestamp is the unified-timeline time (ns) this pose refers to.
	Timestamp int64
	Pose      Pose
	Velocity  Vec3
	State     TrackingState
	// CovarianceTrace is a scalar uncertainty indicator (position
	// covariance trace). It grows while lost and shrinks on fusion.
	CovarianceTrace float64
	// Sequence identifies the estimator revision that produced this
	// snapshot; it is monotonic and useful for reproducibility checks.
	Sequence uint64
}

// RejectEvent records a suppressed observation for diagnostics.
type RejectEvent struct {
	Timestamp int64
	Source    string // "imu" or "vision"
	Reason    RejectReason
}

// Config tunes the tracker. See README for semantics.
type Config struct {
	// Gravity in world frame (m/s^2). Default (0,0,-9.81).
	Gravity Vec3

	// VisionTimeout: no fused vision observation for longer than this
	// marks the tracker lost. Default 500ms.
	VisionTimeout time.Duration
	// RecoveryBlend: duration over which the output is blended from the
	// inertial-only pose to the re-acquired fused pose. Default 300ms.
	RecoveryBlend time.Duration

	// MaxLateVision: vision observations older than this relative to
	// the estimator frontier are dropped as too-late. Default 100ms.
	MaxLateVision time.Duration
	// MaxLateIMU: same for IMU samples. Default 50ms.
	MaxLateIMU time.Duration

	// OutlierGateSigma: innovation gate in units of standard
	// deviations. Default 4.
	OutlierGateSigma float64
	// MaxConsecutiveOutliers: after this many consecutive gated
	// observations the gate is widened once (the stream may have
	// legitimately jumped) instead of letting the estimate freeze.
	// Default 5.
	MaxConsecutiveOutliers int

	// Default noise levels.
	VisionPositionStd    float64 // default 0.02 m
	VisionOrientationStd float64 // default 0.01 rad
	IMUAccelNoise        float64 // default 0.5 m/s^2
	IMUGyroNoise         float64 // default 0.05 rad/s

	// Process noise for random-walk bias models.
	GyroBiasInstability  float64 // default 1e-4 rad/s
	AccelBiasInstability float64 // default 1e-3 m/s^2

	// MaxCovarianceTrace: if the position covariance trace exceeds
	// this, the estimate is declared diverged and reset to the last
	// trusted pose. Default 100 (m^2).
	MaxCovarianceTrace float64

	// HistorySize: number of recent output snapshots kept for
	// interpolated queries. At 230 fused events/s (200 Hz IMU + 30 Hz
	// vision) the default of 8192 covers ~35 s. Queries older than
	// the retained history clamp to the earliest snapshot.
	HistorySize int
}

func (c *Config) withDefaults() Config {
	out := *c
	if out.Gravity == (Vec3{}) {
		out.Gravity = Vec3{Z: -9.81}
	}
	if out.VisionTimeout <= 0 {
		out.VisionTimeout = 500 * time.Millisecond
	}
	if out.RecoveryBlend <= 0 {
		out.RecoveryBlend = 300 * time.Millisecond
	}
	if out.MaxLateVision <= 0 {
		out.MaxLateVision = 100 * time.Millisecond
	}
	if out.MaxLateIMU <= 0 {
		out.MaxLateIMU = 50 * time.Millisecond
	}
	if out.OutlierGateSigma <= 0 {
		out.OutlierGateSigma = 4
	}
	if out.MaxConsecutiveOutliers <= 0 {
		out.MaxConsecutiveOutliers = 5
	}
	if out.VisionPositionStd <= 0 {
		out.VisionPositionStd = 0.02
	}
	if out.VisionOrientationStd <= 0 {
		out.VisionOrientationStd = 0.01
	}
	if out.IMUAccelNoise <= 0 {
		out.IMUAccelNoise = 0.5
	}
	if out.IMUGyroNoise <= 0 {
		out.IMUGyroNoise = 0.05
	}
	if out.GyroBiasInstability <= 0 {
		out.GyroBiasInstability = 1e-4
	}
	if out.AccelBiasInstability <= 0 {
		out.AccelBiasInstability = 1e-3
	}
	if out.MaxCovarianceTrace <= 0 {
		out.MaxCovarianceTrace = 100
	}
	if out.HistorySize <= 0 {
		out.HistorySize = 8192
	}
	return out
}
