package tracking

import "math"

// Quaternion and vector helpers. All functions are pure and
// deterministic.

// qMul returns a*b (apply b first, then a).
func qMul(a, b Quat) Quat {
	return Quat{
		W: a.W*b.W - a.X*b.X - a.Y*b.Y - a.Z*b.Z,
		X: a.W*b.X + a.X*b.W + a.Y*b.Z - a.Z*b.Y,
		Y: a.W*b.Y - a.X*b.Z + a.Y*b.W + a.Z*b.X,
		Z: a.W*b.Z + a.X*b.Y - a.Y*b.X + a.Z*b.W,
	}
}

func qConj(q Quat) Quat { return Quat{W: q.W, X: -q.X, Y: -q.Y, Z: -q.Z} }

func qNormalize(q Quat) Quat {
	n := math.Sqrt(q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z)
	if n == 0 || math.IsNaN(n) {
		return Quat{W: 1}
	}
	return Quat{W: q.W / n, X: q.X / n, Y: q.Y / n, Z: q.Z / n}
}

// qRotate rotates vector v by quaternion q.
func qRotate(q Quat, v Vec3) Vec3 {
	// t = 2 * cross(q.xyz, v)
	tx := 2 * (q.Y*v.Z - q.Z*v.Y)
	ty := 2 * (q.Z*v.X - q.X*v.Z)
	tz := 2 * (q.X*v.Y - q.Y*v.X)
	return Vec3{
		X: v.X + q.W*tx + (q.Y*tz - q.Z*ty),
		Y: v.Y + q.W*ty + (q.Z*tx - q.X*tz),
		Z: v.Z + q.W*tz + (q.X*ty - q.Y*tx),
	}
}

func qFromAxisAngle(axis Vec3, angle float64) Quat {
	n := vNorm(axis)
	if n == 0 {
		return Quat{W: 1}
	}
	h := angle * 0.5
	s := math.Sin(h) / n
	return Quat{W: math.Cos(h), X: axis.X * s, Y: axis.Y * s, Z: axis.Z * s}
}

// qFromRotVec converts a rotation vector (axis*angle) to a quaternion.
func qFromRotVec(rv Vec3) Quat {
	angle := vNorm(rv)
	if angle < 1e-12 {
		// First-order approximation to avoid division by zero.
		return qNormalize(Quat{W: 1, X: rv.X * 0.5, Y: rv.Y * 0.5, Z: rv.Z * 0.5})
	}
	return qFromAxisAngle(rv, angle)
}

// qToRotVec converts a quaternion to a rotation vector.
func qToRotVec(q Quat) Vec3 {
	q = qNormalize(q)
	if q.W < 0 { // shortest path
		q = Quat{W: -q.W, X: -q.X, Y: -q.Y, Z: -q.Z}
	}
	sinHalf := math.Sqrt(q.X*q.X + q.Y*q.Y + q.Z*q.Z)
	if sinHalf < 1e-12 {
		return Vec3{X: 2 * q.X, Y: 2 * q.Y, Z: 2 * q.Z}
	}
	angle := 2 * math.Atan2(sinHalf, q.W)
	k := angle / sinHalf
	return Vec3{X: q.X * k, Y: q.Y * k, Z: q.Z * k}
}

// qSlerp interpolates between a and b (t in [0,1]).
func qSlerp(a, b Quat, t float64) Quat {
	dot := a.W*b.W + a.X*b.X + a.Y*b.Y + a.Z*b.Z
	if dot < 0 {
		b = Quat{W: -b.W, X: -b.X, Y: -b.Y, Z: -b.Z}
		dot = -dot
	}
	if dot > 0.9995 {
		return qNormalize(Quat{
			W: a.W + t*(b.W-a.W),
			X: a.X + t*(b.X-a.X),
			Y: a.Y + t*(b.Y-a.Y),
			Z: a.Z + t*(b.Z-a.Z),
		})
	}
	theta := math.Acos(math.Min(1, dot))
	s := math.Sin(theta)
	wa := math.Sin((1-t)*theta) / s
	wb := math.Sin(t*theta) / s
	return Quat{
		W: wa*a.W + wb*b.W,
		X: wa*a.X + wb*b.X,
		Y: wa*a.Y + wb*b.Y,
		Z: wa*a.Z + wb*b.Z,
	}
}

func vAdd(a, b Vec3) Vec3           { return Vec3{X: a.X + b.X, Y: a.Y + b.Y, Z: a.Z + b.Z} }
func vSub(a, b Vec3) Vec3           { return Vec3{X: a.X - b.X, Y: a.Y - b.Y, Z: a.Z - b.Z} }
func vScale(a Vec3, s float64) Vec3 { return Vec3{X: a.X * s, Y: a.Y * s, Z: a.Z * s} }
func vDot(a, b Vec3) float64        { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func vNorm(a Vec3) float64          { return math.Sqrt(vDot(a, a)) }

func isFiniteVec(v Vec3) bool {
	return !math.IsNaN(v.X) && !math.IsNaN(v.Y) && !math.IsNaN(v.Z) &&
		!math.IsInf(v.X, 0) && !math.IsInf(v.Y, 0) && !math.IsInf(v.Z, 0)
}

func isFiniteQuat(q Quat) bool {
	return !math.IsNaN(q.W) && !math.IsNaN(q.X) && !math.IsNaN(q.Y) && !math.IsNaN(q.Z) &&
		!math.IsInf(q.W, 0) && !math.IsInf(q.X, 0) && !math.IsInf(q.Y, 0) && !math.IsInf(q.Z, 0)
}
