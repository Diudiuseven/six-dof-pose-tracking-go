package tracking

// filter.go: the fusion estimator. It propagates a strapdown inertial
// state between absolute observations and applies gated complementary
// corrections from vision. Missing observations are simply absent: the
// estimator never fabricates zero-valued measurements. All arithmetic is
// deterministic; identical event sequences give identical states.

// estimator is the fused navigation state at a point on the unified
// timeline. It is a plain value: checkpoints copy it by value.
type estimator struct {
	time         int64
	pos          Vector3
	vel          Vector3
	ori          Quaternion
	gyroBias     Vector3
	lastVisionAt int64 // unified time of the last accepted vision update
	initialized  bool
}

func newEstimator() estimator {
	return estimator{ori: Quaternion{W: 1}}
}

// initFrom seeds the estimator from the first absolute observation.
func (e *estimator) initFrom(o visionEntry) {
	e.time = o.Time
	e.pos = o.Pose.Position
	e.ori = qNormalize(o.Pose.Orientation)
	e.vel = Vector3{}
	e.lastVisionAt = o.Time
	e.initialized = true
}

// propagate integrates one IMU sample. dt is derived from entry
// timestamps so out-of-order insertion followed by replay stays exact.
func (e *estimator) propagate(imu imuEntry, gravity Vector3) {
	if !e.initialized {
		// Before the first absolute observation we still integrate
		// relative motion from identity so early queries return
		// something coherent, marked untrusted by the caller.
		e.initialized = true
		e.time = imu.Time
	}
	dt := float64(imu.Time-e.time) * 1e-9
	if dt <= 0 {
		e.time = imu.Time
		return
	}
	// Orientation: integrate bias-corrected angular rate.
	w := vSub(imu.Gyro, e.gyroBias)
	e.ori = qNormalize(qMul(e.ori, qFromAxisAngle(vScale(w, dt))))
	// Translation: specific force rotated to world, plus gravity.
	a := vAdd(qRotate(e.ori, imu.Accel), gravity)
	e.pos = vAdd(e.pos, vAdd(vScale(e.vel, dt), vScale(a, 0.5*dt*dt)))
	e.vel = vAdd(e.vel, vScale(a, dt))
	e.time = imu.Time
}

// innovation measures how far an observation is from the current
// prediction: position distance in meters plus rotation angle in
// radians. Used for outlier gating.
func (e *estimator) innovation(o visionEntry) float64 {
	dp := vNorm(vSub(o.Pose.Position, e.pos))
	dr := vNorm(qLog(qMul(qConj(e.ori), o.Pose.Orientation)))
	return dp + dr
}

// gate returns the acceptance threshold for an observation, scaled by
// its reported uncertainty.
func gateThreshold(cfg Config, covariance float64) float64 {
	c := covariance
	if c < 0 {
		c = 0
	}
	return cfg.OutlierGate * (1 + c)
}

// correct fuses an accepted vision observation with a complementary
// update. The trust weight k shrinks as the reported covariance grows.
// It also updates the gyro bias estimate so inertial-only stretches do
// not drift without bound while vision is intermittently available.
func (e *estimator) correct(o visionEntry, cfg Config) {
	k := 1 / (1 + o.Covariance)
	posErr := vSub(o.Pose.Position, e.pos)
	rotErr := qLog(qMul(qConj(e.ori), o.Pose.Orientation))

	// Slow integral on the rotation error estimates the gyro bias.
	e.gyroBias = vAdd(e.gyroBias, vScale(rotErr, cfg.GyroBiasAlpha))

	// Velocity correction from the position error over the interval
	// since the previous vision update keeps velocity bounded.
	if dt := float64(o.Time-e.lastVisionAt) * 1e-9; dt > 1e-3 {
		e.vel = vAdd(e.vel, vScale(posErr, k*0.5/dt))
	}

	e.pos = vLerp(e.pos, o.Pose.Position, k)
	e.ori = qNormalize(qSlerp(e.ori, o.Pose.Orientation, k))
	e.lastVisionAt = o.Time
}

// finite reports whether the state is numerically sane.
func (e *estimator) finite() bool {
	return vFinite(e.pos) && vFinite(e.vel) && qFinite(e.ori) && vFinite(e.gyroBias)
}
