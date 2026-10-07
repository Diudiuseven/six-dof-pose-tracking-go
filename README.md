# six-dof-pose-tracking-go

面向头显与移动设备的六自由度（6-DoF）位姿追踪运行时。将高频惯性测量（IMU）
与低频绝对视觉观测融合到**统一时间轴**上，提供可对齐到任意请求时刻的位姿查询，
并内建校时、乱序重处理、异常抑制、丢失恢复与数值安全能力。

```
import "github.com/Diudiuseven/six-dof-pose-tracking-go/tracking"

tr := tracking.NewTracker(tracking.Config{}) // 零值配置即合理默认

// 传感器线程（可并发）
tr.IngestIMU(sample, unifiedNow)        // 200Hz+
reason := tr.IngestVision(obs, unifiedNow) // ~30Hz

// 渲染/查询线程（可并发）
snap := tr.PoseAt(renderTime) // 对齐到请求时刻的自洽快照
```

## 时间模型

- **统一时间轴**：单调递增的 int64 纳秒。所有输出（`PoseSnapshot.Timestamp`）
  都在此时间轴上。调用方在每次喂入数据时传入当前的统一时刻 `unifiedNow`。
- **传感器时钟**：每路传感器（`ClockIMU` / `ClockVision`）有自己的时钟域，
  时间戳可以带任意固定偏移并允许缓慢漂移。运行时为每路时钟维护一个
  `clockAligner`：用 `(sensorTime, unifiedNow)` 观测对以有界指数滑动平均
  估计偏移量（单次修正上限 5ms，防止延迟突发引起时间轴跳变），
  将传感器时间戳换算到统一时间轴后再参与融合。
- **对齐查询**：`PoseAt(t)` 不返回"最近一次采样"，而是在输出历史中按时间
  二分查找并做线性/球面插值；超出最新时刻时按速度外推，外推上限 100ms；
  早于保留历史时钳到最早快照。历史长度由 `HistorySize` 控制
  （默认 8192 条，约覆盖 35s）。

## 融合语义

估计器为误差状态扩展卡尔曼滤波（ES-EKF）：

- **预测**：每条 IMU 样本按实际 `dt` 积分（陀螺去偏积分姿态，加计去偏并
  扣除重力后积分位置/速度）。位置-速度块使用精确的运动学离散化协方差传播
  （含互协方差项），姿态与零偏按随机游走模型增长不确定性。
  **缺失的样本不会被补零**，只是使下一步积分的 `dt` 变大。
- **更新**：视觉观测提供绝对位姿。位置/速度按逐轴 2×2 标量卡尔曼增益更新
  （速度通过位置-速度互协方差被间接校正），姿态按旋转向量误差更新；
  陀螺/加计零偏以小固定增益向与残差一致的方向松弛，保证长时间运行
  误差无界增长被抑制。
- **乱序与迟到**：事件按统一时间戳插入有序日志。落在重处理窗口
  （`MaxLateVision` / `MaxLateIMU`）内的迟到事件触发**回滚-重放**：
  恢复到插入点之前的检查点，按序重放窗口内所有事件，结果与按序到达一致；
  超出窗口的按 `RejectTooLate` 丢弃。
- **重复**：同一来源同一时间戳的观测只计入一次，重复到达返回
  `RejectDuplicate`，不会重复积分。
- **异常抑制**：观测先经有限性检查（`RejectNonFinite`），再做马氏距离门限
  （`OutlierGateSigma`σ）。被门限拦截的观测以
  `RejectOutlierPosition` / `RejectOutlierOrientation` 记入诊断日志
  （`Rejects()`），不污染估计。连续异常超过 `MaxConsecutiveOutliers` 时
  执行一次**有界追赶**（向测量方向移动 50% 并膨胀协方差），
  使真实跳变场景能恢复，同时估计不会冻结或发散。
- **丢失与恢复**：超过 `VisionTimeout` 没有融合视觉观测即进入 `Lost`，
  期间仅靠惯性推进，快照的 `State` 明确标记为不可信。重新捕获绝对观测后
  进入 `Recovering`：输出不跳变——以重捕获瞬间的输出位姿与融合位姿之差
  作为衰减偏移，在 `RecoveryBlend` 时长内线性地从纯惯性输出平滑混合回
  融合估计，随后回到 `Tracking`。
- **数值安全**：每个事件处理后检查协方差迹上限（`MaxCovarianceTrace`）与
  NaN/Inf；一旦发散，以当前（或最后可信）位姿重置滤波器并标记 `Lost`，
  绝不把不可用结果当作正常输出。

## 并发模型

写入（`IngestIMU` / `IngestVision`）持写锁，临界区仅为单个事件的处理；
查询（`PoseAt` / `State` / `Rejects`）持读锁，多查询可并行。`PoseSnapshot`
为值类型，任一时刻返回的都是某一估计版本完整、自洽的快照（姿态、速度、
状态、不确定度来自同一版本），不存在字段级混合。

## 可复现性

估计路径无任何随机数与墙钟读取：时钟对齐、融合、门限、混合全部由输入
序列决定。固定随机种子的输入序列在两次运行中产生**逐条一致**的输出快照
（见 `TestDeterministicReproduction`）。

## 配置项（Config）

| 字段 | 默认 | 含义 |
|---|---|---|
| `Gravity` | (0,0,-9.81) | 世界系重力矢量 (m/s²) |
| `VisionTimeout` | 500ms | 超过此时长无视觉融合即判定丢失 |
| `RecoveryBlend` | 300ms | 重捕获后输出平滑混合时长 |
| `MaxLateVision` | 100ms | 视觉观测可重处理的最大迟到窗口 |
| `MaxLateIMU` | 50ms | IMU 样本可重处理的最大迟到窗口 |
| `OutlierGateSigma` | 4 | 异常门限（标准差倍数） |
| `MaxConsecutiveOutliers` | 5 | 触发有界追赶的连续异常数 |
| `VisionPositionStd` | 0.02m | 视觉位置噪声默认 1σ |
| `VisionOrientationStd` | 0.01rad | 视觉姿态噪声默认 1σ |
| `IMUAccelNoise` | 0.5 m/s² | 加计噪声谱密度 |
| `IMUGyroNoise` | 0.05 rad/s | 陀螺噪声谱密度 |
| `GyroBiasInstability` | 1e-4 rad/s | 陀螺零偏随机游走 |
| `AccelBiasInstability` | 1e-3 m/s² | 加计零偏随机游走 |
| `MaxCovarianceTrace` | 100 m² | 位置协方差迹发散阈值 |
| `HistorySize` | 8192 | 输出历史快照条数 |

单条观测可通过 `VisionObservation.PositionStd / OrientationStd`
覆盖默认噪声（传 0 用默认值）。

## 测试

```
go test ./...        # 功能测试
go test -race ./...  # 并发竞态检测
```

覆盖：确定性复现、真值跟踪精度、乱序重处理、迟到丢弃、重复去重、
异常抑制与连续异常不发散、追踪丢失与平滑重捕获、120s 长跑漂移有界、
非有限输入拒绝、并发查询一致性、跨采样插值对齐。
