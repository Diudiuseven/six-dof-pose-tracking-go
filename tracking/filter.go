package tracking

import "math"

// Error-state indices into the diagonal covariance array.
const (
	errPos   = 0 // 0,1,2
	errVel   = 3 // 3,4,5
	errTheta = 6 // 6,7,8
	errGyroB = 9 // 9,10,11
	errAccB  = 12
)

// filterState is the full navigation state.
type filterState struct {
	pose     Pose
	vel      Vec3
	gyroBias Vec3
	accBias  Vec3
	// Diagonal covariance of the 15-dim error state, plus the
	// position-velocity cross covariance per axis (the only
	// off-diagonal terms the prediction model creates). This keeps
	// the math exact for the kinematic chain while remaining
	// numerically robust and deterministic.
	cov     [15]float64
	pvCross [3]float64
}

// filter is an error-state EKF fusing IMU prediction with absolute
// vision pose updates.
type filter struct {
	cfg Config
	st  filterState

	consecutiveOutliers int
}

func newFilter(cfg Config) *filter {
	f := &filter{cfg: cfg}
	f.reset(Pose{Orientation: Quat{W: 1}})
	return f
}

func (f *filter) reset(p Pose) {
	gb, ab := f.st.gyroBias, f.st.accBias
	f.st = filterState{pose: p, gyroBias: gb, accBias: ab}
	for i := 0; i < 3; i++ {
		f.st.cov[errPos+i] = 1.0
		f.st.cov[errVel+i] = 1.0
		f.st.cov[errTheta+i] = 0.1
		f.st.cov[errGyroB+i] = 1e-4
		f.st.cov[errAccB+i] = 1e-2
	}
	f.consecutiveOutliers = 0
}

// predict advances the state by one IMU sample over dt seconds.
func (f *filter) predict(s IMUSample, dt float64) {
	if dt <= 0 {
		return
	}
	st := &f.st

	w := vSub(s.Gyro, st.gyroBias)
	aBody := vSub(s.Accel, st.accBias)

	// Orientation: q <- q * exp(w*dt).
	st.pose.Orientation = qNormalize(qMul(st.pose.Orientation, qFromRotVec(vScale(w, dt))))

	// Position/velocity with average acceleration over the interval.
	aWorld := vAdd(qRotate(st.pose.Orientation, aBody), f.cfg.Gravity)
	st.pose.Position = vAdd(st.pose.Position,
		vAdd(vScale(st.vel, dt), vScale(aWorld, 0.5*dt*dt)))
	st.vel = vAdd(st.vel, vScale(aWorld, dt))

	// Covariance propagation. The position/velocity block uses the
	// exact kinematic discretization with white acceleration noise
	// of spectral density qa; orientation and biases are simple
	// random walks.
	qa := f.cfg.IMUAccelNoise * f.cfg.IMUAccelNoise
	qg := f.cfg.IMUGyroNoise * f.cfg.IMUGyroNoise * dt
	for i := 0; i < 3; i++ {
		p := st.cov[errPos+i]
		v := st.cov[errVel+i]
		c := st.pvCross[i]
		st.cov[errPos+i] = p + 2*c*dt + v*dt*dt + qa*dt*dt*dt/3
		st.pvCross[i] = c + v*dt + qa*dt*dt/2
		st.cov[errVel+i] = v + qa*dt
		st.cov[errTheta+i] += qg
		st.cov[errGyroB+i] += f.cfg.GyroBiasInstability * f.cfg.GyroBiasInstability * dt
		st.cov[errAccB+i] += f.cfg.AccelBiasInstability * f.cfg.AccelBiasInstability * dt
	}
}

func trace3(c [15]float64, base int) float64 {
	return c[base] + c[base+1] + c[base+2]
}

func compOf(v Vec3, i int) float64 {
	switch i {
	case 0:
		return v.X
	case 1:
		return v.Y
	}
	return v.Z
}

func setComp(v *Vec3, i int, x float64) {
	switch i {
	case 0:
		v.X = x
	case 1:
		v.Y = x
	default:
		v.Z = x
	}
}

// updateVision fuses an absolute pose observation. Returns the
// reject reason (RejectNone on success).
func (f *filter) updateVision(o VisionObservation) RejectReason {
	st := &f.st

	posStd := o.PositionStd
	if posStd <= 0 {
		posStd = f.cfg.VisionPositionStd
	}
	oriStd := o.OrientationStd
	if oriStd <= 0 {
		oriStd = f.cfg.VisionOrientationStd
	}
	rPos := posStd * posStd
	rOri := oriStd * oriStd

	// Innovation.
	innovPos := vSub(o.Pose.Position, st.pose.Position)
	innovOri := qToRotVec(qMul(o.Pose.Orientation, qConj(st.pose.Orientation)))

	// Gating: Mahalanobis-like test.
	gate := f.cfg.OutlierGateSigma
	posVar := trace3(st.cov, errPos) + 3*rPos
	oriVar := trace3(st.cov, errTheta) + 3*rOri
	if vDot(innovPos, innovPos) > gate*gate*posVar {
		return f.rejectOrCatchUp(RejectOutlierPosition, o)
	}
	if vDot(innovOri, innovOri) > gate*gate*oriVar {
		return f.rejectOrCatchUp(RejectOutlierOrientation, o)
	}

	// Position/velocity update per axis: exact scalar Kalman on the
	// 2x2 (pos, vel) block.
	for i := 0; i < 3; i++ {
		p := st.cov[errPos+i]
		v := st.cov[errVel+i]
		c := st.pvCross[i]
		s := p + rPos
		kp := p / s
		kv := c / s
		innov := compOf(innovPos, i)

		setComp(&st.pose.Position, i, compOf(st.pose.Position, i)+kp*innov)
		setComp(&st.vel, i, compOf(st.vel, i)+kv*innov)

		st.cov[errPos+i] = p - kp*p
		st.pvCross[i] = c - kp*c
		st.cov[errVel+i] = v - kv*c
	}

	// Orientation update per axis (diagonal).
	for i := 0; i < 3; i++ {
		p := st.cov[errTheta+i]
		ko := p / (p + rOri)
		setComp(&innovOri, i, compOf(innovOri, i)*ko)
		st.cov[errTheta+i] = p * (1 - ko)
	}
	st.pose.Orientation = qNormalize(qMul(qFromRotVec(innovOri), st.pose.Orientation))

	// Bias relaxation: small fixed gains pull biases toward values
	// consistent with the observed correction. Bounded and
	// deterministic; keeps long-run drift from growing without limit.
	const kGyroBias = 0.02
	const kAccBias = 0.01
	st.gyroBias = vAdd(st.gyroBias, vScale(qRotate(qConj(st.pose.Orientation), innovOri), -kGyroBias))
	st.accBias = vAdd(st.accBias, vScale(qRotate(qConj(st.pose.Orientation), innovPos), kAccBias))

	f.consecutiveOutliers = 0
	return RejectNone
}

// rejectOrCatchUp counts consecutive outliers; if the stream appears
// to have legitimately jumped (persistent "outliers"), the filter
// performs a bounded catch-up toward the measurement instead of
// freezing forever.
func (f *filter) rejectOrCatchUp(reason RejectReason, o VisionObservation) RejectReason {
	f.consecutiveOutliers++
	if f.consecutiveOutliers > f.cfg.MaxConsecutiveOutliers {
		// Bounded catch-up: move a fraction toward the measurement
		// and inflate covariance so subsequent updates fuse normally.
		const alpha = 0.5
		f.st.pose.Position = vAdd(f.st.pose.Position,
			vScale(vSub(o.Pose.Position, f.st.pose.Position), alpha))
		f.st.pose.Orientation = qSlerp(f.st.pose.Orientation, o.Pose.Orientation, alpha)
		for i := 0; i < 3; i++ {
			f.st.cov[errPos+i] += 0.5
			f.st.cov[errTheta+i] += 0.05
		}
		f.consecutiveOutliers = 0
	}
	return reason
}

// diverged reports whether the estimate exceeded uncertainty limits
// or became non-finite.
func (f *filter) diverged() bool {
	if trace3(f.st.cov, errPos) > f.cfg.MaxCovarianceTrace {
		return true
	}
	st := &f.st
	if !isFiniteVec(st.pose.Position) || !isFiniteVec(st.vel) ||
		!isFiniteQuat(st.pose.Orientation) ||
		!isFiniteVec(st.gyroBias) || !isFiniteVec(st.accBias) {
		return true
	}
	for _, c := range st.cov {
		if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
			return true
		}
	}
	for _, c := range st.pvCross {
		if math.IsNaN(c) || math.IsInf(c, 0) {
			return true
		}
	}
	return false
}
