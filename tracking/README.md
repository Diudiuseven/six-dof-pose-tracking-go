# six-dof-pose-tracking-go

面向头显与移动设备的六自由度（6-DOF）位姿追踪运行时。它将高频惯性测量（IMU）
与低频绝对视觉观测融合到**统一时间轴**上，对外提供可对齐到任意请求时刻的位姿查询，
并在一处收敛校时、融合、乱序处理、异常抑制与追踪丢失恢复能力。

核心包：`tracking`（纯 Go，无外部依赖）。

## 快速开始

```go
cfg := tracking.DefaultConfig()
cfg.ClockOffsetVision = -40_000_000 // 视觉时钟比统一时钟快 40ms
tr := tracking.NewTracker(cfg)

// 写入侧（可来自不同 goroutine）：
tr.PushIMU(tracking.IMUSample{Seq: 1, SensorTime: tIMU, Accel: a, Gyro: g})
tr.PushVision(tracking.VisionObservation{Seq: 1, SensorTime: tVis, Pose: p, Covariance: 0.01})

// 查询侧（渲染线程，任意时刻）：
if s, ok := tr.PoseAt(renderTime); ok {
    // s 是一个自洽的完整快照：Time/Pose/State/Quality
}
```

## 时间模型

- **统一时间轴**：所有输出（`PoseSample.Time`、`PoseAt` 的查询参数）都在同一条
  纳秒时间轴上。
- **传感器时钟映射**：每路传感器使用自己的时钟采样。转换关系为
  `unified = sensorTime + offset`，偏移量由 `Config.ClockOffsetIMU /
  ClockOffsetVision` 给出（例如由出厂标定或校时协议测得）。
- **对齐输出**：`PoseAt(t)` 在保留的快照环内做插值（位置 lerp、姿态 slerp），
  因此输出对齐到**请求的时刻**，而不是最近一次采样。查询时刻超出保留范围时，
  钳位到最近可用快照。
- **迟到观测**：时间戳落在历史窗口（`HistoryWindow`）内的迟到观测会被插入其
  正确位置，并从最近的检查点**确定性重放**受影响的时间线后缀，效果等同于它
  按时到达；早于窗口的观测按 `RejectTooLate` 丢弃。

## 融合语义

- **惯性递推**：陀螺（扣除在线估计的偏置）积分姿态，比力转换到世界系后加重力
  二次积分位置。缺失的观测不参与融合——估计器从不把缺省/零值当作测量。
- **视觉校正**：互补滤波。信任权重 `k = 1/(1+Covariance)`，同时校正位置、姿态、
  速度，并以慢积分（`GyroBiasAlpha`）估计陀螺偏置，保证长时间运行误差有界。
- **乱序/重复**：每路流按序列号去重（`RejectDuplicate`）；同一时间戳的不同样本
  按 `RejectStaleTimestamp` 拒绝；乱序样本插入正确位置后重放。
- **异常抑制**：新观测先过门控——归一化新息（位置距离 + 旋转角）超过
  `OutlierGate*(1+Covariance)` 即被抑制，`PushVision` 返回 `RejectOutlier`，
  估计不受影响。连续被抑制超过 `MaxConsecutiveOutliers` 次时，视为场景真实变化，
  强制接受并通过恢复平滑机制吸收跳变，避免估计卡死在错误坐标系上。
- **数值安全**：含 NaN/Inf 的输入按 `RejectInvalid` 拒绝；积分后状态若出现非有限
  值，该样本被丢弃、状态回退，不会把不可用结果当作正常输出。

## 追踪丢失与恢复

状态机（`PoseSample.State`，配合 `Quality ∈ [0,1]`）：

| 状态 | 含义 | Quality |
|---|---|---|
| `tracking` | 视觉+惯性正常融合 | 1.0 |
| `degraded` | 视觉间歇（超过 LostTimeout/2 未收到） | 0.6 |
| `lost` | 超过 `LostTimeout` 无绝对观测，纯惯性外推，**不可信** | 0.0 |
| `recovering` | 重新捕获绝对观测，输出正在平滑混合回绝对坐标系 | 0.85 |

重新捕获时，校正量不会直接跳到输出上：系统记录捕获瞬间的"已发布位姿"，
把校正差值吸收进一个随 `RecoveryBlend` 时长线性衰减的输出偏移中，位姿输出
连续、无跳变，且整个过程可通过状态与 Quality 解释。

## 并发与一致性

- 写（`PushIMU`/`PushVision`）与读（`PoseAt`/`Latest`）可并发。
- 读者只读取发布后不可变的 `PoseSample` 快照环，任一时刻查询到的都是某个
  自洽的完整快照（或两个快照间的插值），不存在字段级混合。
- 读路径只持有读锁，写入侧不会因查询而长时间阻塞。

## 确定性

追踪器内部无任何随机源与墙钟依赖：在固定随机种子生成的相同输入序列
（含相同的到达顺序）下，估计结果与输出时间轴逐条一致，可复现历史结论。
`TestDeterminism` 对此做了逐快照精确相等验证。

## 配置项（Config）

| 字段 | 默认 | 说明 |
|---|---|---|
| `ClockOffsetIMU` / `ClockOffsetVision` | 0 | 传感器时钟到统一时间轴的偏移（ns） |
| `HistoryWindow` | 2s | 可接受迟到观测与重放的历史窗口（ns） |
| `LostTimeout` | 500ms | 无视觉观测多久后进入 `lost`（ns） |
| `RecoveryBlend` | 300ms | 重捕获后输出平滑混合时长（ns） |
| `OutlierGate` | 1.5 | 异常门控阈值（米+弧度，按协方差缩放） |
| `MaxConsecutiveOutliers` | 5 | 连续异常多少次后强制接受（场景切换策略） |
| `GyroBiasAlpha` | 0.001 | 陀螺偏置在线估计学习率 |
| `Gravity` | (0,0,-9.81) | 世界系重力向量 |

## 测试

```sh
go test ./...        # 单元与场景测试
go test -race ./...  # 并发查询竞态检测
```

覆盖：乱序与迟到、重复与陈旧时间戳、异常观测抑制、追踪丢失与平滑重捕获、
长时间漂移有界性、并发查询、固定种子下的逐条确定性，以及传感器时钟偏移补偿。
