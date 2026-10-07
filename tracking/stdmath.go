package tracking

// Thin wrappers over the standard library so the math helpers above stay
// one-line and the whole package has a single import site for math.
import "math"

func sqrt(v float64) float64     { return math.Sqrt(v) }
func sin(v float64) float64      { return math.Sin(v) }
func cos(v float64) float64      { return math.Cos(v) }
func acos(v float64) float64     { return math.Acos(v) }
func atan2(y, x float64) float64 { return math.Atan2(y, x) }
func isNaN(v float64) bool       { return math.IsNaN(v) }
func isInf(v float64) bool       { return math.IsInf(v, 0) }
