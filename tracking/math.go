package tracking

// math.go: small deterministic vector/quaternion helpers. No randomness,
// no globals; every function is pure.

func vAdd(a, b Vector3) Vector3 { return Vector3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func vSub(a, b Vector3) Vector3 { return Vector3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func vScale(a Vector3, s float64) Vector3 {
	return Vector3{a.X * s, a.Y * s, a.Z * s}
}
func vDot(a, b Vector3) float64 { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func vNorm(a Vector3) float64   { return sqrt(vDot(a, a)) }
func vLerp(a, b Vector3, t float64) Vector3 {
	return Vector3{a.X + (b.X-a.X)*t, a.Y + (b.Y-a.Y)*t, a.Z + (b.Z-a.Z)*t}
}

func qMul(a, b Quaternion) Quaternion {
	return Quaternion{
		W: a.W*b.W - a.X*b.X - a.Y*b.Y - a.Z*b.Z,
		X: a.W*b.X + a.X*b.W + a.Y*b.Z - a.Z*b.Y,
		Y: a.W*b.Y - a.X*b.Z + a.Y*b.W + a.Z*b.X,
		Z: a.W*b.Z + a.X*b.Y - a.Y*b.X + a.Z*b.W,
	}
}

func qConj(q Quaternion) Quaternion { return Quaternion{W: q.W, X: -q.X, Y: -q.Y, Z: -q.Z} }

func qNorm(q Quaternion) float64 {
	return sqrt(q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z)
}

func qNormalize(q Quaternion) Quaternion {
	n := qNorm(q)
	if n == 0 {
		return Quaternion{W: 1}
	}
	return Quaternion{W: q.W / n, X: q.X / n, Y: q.Y / n, Z: q.Z / n}
}

// qRotate rotates vector v by quaternion q.
func qRotate(q Quaternion, v Vector3) Vector3 {
	qv := Quaternion{W: 0, X: v.X, Y: v.Y, Z: v.Z}
	r := qMul(qMul(q, qv), qConj(q))
	return Vector3{r.X, r.Y, r.Z}
}

// qFromAxisAngle builds a quaternion from a rotation vector (axis*angle).
func qFromAxisAngle(rv Vector3) Quaternion {
	angle := vNorm(rv)
	if angle < 1e-12 {
		return Quaternion{W: 1, X: rv.X / 2, Y: rv.Y / 2, Z: rv.Z / 2}
	}
	s := sin(angle/2) / angle
	return qNormalize(Quaternion{W: cos(angle / 2), X: rv.X * s, Y: rv.Y * s, Z: rv.Z * s})
}

// qLog returns the rotation vector (axis*angle) of a unit quaternion.
func qLog(q Quaternion) Vector3 {
	q = qNormalize(q)
	if q.W < 0 {
		q = Quaternion{-q.W, -q.X, -q.Y, -q.Z}
	}
	v := Vector3{q.X, q.Y, q.Z}
	s := vNorm(v)
	if s < 1e-12 {
		return Vector3{2 * q.X, 2 * q.Y, 2 * q.Z}
	}
	angle := 2 * atan2(s, q.W)
	return vScale(v, angle/s)
}

// qSlerp interpolates between unit quaternions a and b; t in [0,1].
func qSlerp(a, b Quaternion, t float64) Quaternion {
	d := a.W*b.W + a.X*b.X + a.Y*b.Y + a.Z*b.Z
	if d < 0 {
		b = Quaternion{-b.W, -b.X, -b.Y, -b.Z}
		d = -d
	}
	if d > 0.9995 {
		return qNormalize(Quaternion{
			W: a.W + (b.W-a.W)*t, X: a.X + (b.X-a.X)*t,
			Y: a.Y + (b.Y-a.Y)*t, Z: a.Z + (b.Z-a.Z)*t,
		})
	}
	theta0 := acos(d)
	theta := theta0 * t
	sinT := sin(theta)
	sinT0 := sin(theta0)
	s0 := cos(theta) - d*sinT/sinT0
	s1 := sinT / sinT0
	return Quaternion{
		W: s0*a.W + s1*b.W, X: s0*a.X + s1*b.X,
		Y: s0*a.Y + s1*b.Y, Z: s0*a.Z + s1*b.Z,
	}
}

// poseLerp interpolates a full pose (lerp position, slerp orientation).
func poseLerp(a, b Pose, t float64) Pose {
	return Pose{
		Position:    vLerp(a.Position, b.Position, t),
		Orientation: qSlerp(a.Orientation, b.Orientation, t),
	}
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func isFiniteF(v float64) bool { return !isNaN(v) && !isInf(v) }

func vFinite(v Vector3) bool { return isFiniteF(v.X) && isFiniteF(v.Y) && isFiniteF(v.Z) }

func qFinite(q Quaternion) bool {
	return isFiniteF(q.W) && isFiniteF(q.X) && isFiniteF(q.Y) && isFiniteF(q.Z)
}
