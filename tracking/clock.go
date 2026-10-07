package tracking

// clock.go: maps per-sensor clocks onto the unified timeline.

// clockMap converts sensor timestamps to unified timestamps.
type clockMap struct {
	offset int64
}

func newClockMap(offset int64) *clockMap { return &clockMap{offset: offset} }

// toUnified converts a sensor timestamp to the unified timeline.
func (c *clockMap) toUnified(sensorTime int64) int64 { return sensorTime + c.offset }
