# LIBERO 吸盘数据两阶段再生成说明

这个目录放的是“吸盘版 LIBERO 数据再生成”流程的入口脚本和共享实时控制模块。整体思路不是直接改官方一个脚本，而是拆成两阶段：

1. 第一阶段先采“真实执行”的原始 HDF5。
2. 第二阶段再按官方 `regenerate_libero_dataset.py` 的风格做回放、删 `no-op`、删失败轨迹。

当前实时交互链路已经统一成：

- 浏览器页面通过 WebSocket 把按钮点击和键盘事件发给服务器
- 服务器通过 FastAPI + WebSocket 把视频、阶段说明、按钮说明、模式状态和六维力推回浏览器

## 文件说明

- `collect_libero_suction_handover_dataset.py`
  - 第一阶段原始采集脚本。
  - 浏览器页面流程已经不是“一启动就直接采集”，而是：
    1. 进入未完成 task 后，先显示 `播放原始LIBERO轨迹 / 退出`
    2. 点击 `播放原始LIBERO轨迹` 后，自动寻找当前 task 第一条成功源 demo，并完整预览一遍；如果没找到成功 demo，就 fallback 到 `demo_0`
    3. 预览结束后，显示 `数据采集 / 重播 / 退出`
    4. 每次点击 `数据采集`，只采当前 task 的下一个未完成 demo
  - 真实采集时仍然采用：
    - 自动前段沿用原始轨迹
    - `H` 触发 `自动 -> 人工`
    - `R` 重启当前 demo，丢弃当前缓存并回到 `数据采集 / 重播 / 退出`
    - `Q` 或 `ESC` 退出整个脚本
  - 这个脚本会完整记录真实执行动作、图像、状态、`suction_wrench`、吸盘状态等信息。
  - 这个脚本**不会**删除 `no-op`，也**不会**筛掉失败轨迹。

- `regenerate_suction_libero_dataset.py`
  - 第二阶段清洗与再生成脚本。
  - 输入是第一阶段产出的原始 HDF5。
  - 它会像官方 `experiments/robot/libero/regenerate_libero_dataset.py` 那样，在环境里重新回放这些轨迹，重新采样观测，删除 `no-op`，并删除最终失败的整条轨迹。
  - 输出是更适合后续训练使用的干净版 HDF5。
  - 这个脚本保持离线，不接入 WebSocket，因为它不承担人工交互职责。

- `realtime_control_and_monitor.py`
  - 共享工具模块。
  - 封装了浏览器按钮、浏览器键盘输入状态管理，以及 WebSocket 双向通信。
  - `LIBERO/web_teleop.py` 和第一阶段采集脚本都会复用这套工具层。

## 第一阶段的断点续采规则

第一阶段采集脚本的输出目录固定为：

- `/data1/ps/openvla/data/{任务套件名}`

例如：

- `libero_spatial` 对应 `/data1/ps/openvla/data/libero_spatial`

这个目录里：

- 每个 task 对应一个 `{task.name}_demo.hdf5`
- 还会有一个 `{task_suite}_raw_collection_metainfo.json`

再次运行第一阶段脚本时，它会先扫描这些输出文件：

- 如果某个 task 的输出 HDF5 已经包含了和源 HDF5 一样数量的 `demo_x`，就认定这个 task 已完成，直接跳过
- 如果某个 task 只采到一部分，就从下一个缺失 demo 继续
- 如果同时存在“半完成 task”和“完全没开始的 task”，脚本会优先继续半完成 task，避免重头开始浪费时间

## 为什么要拆成两阶段

原始 LIBERO demo 是按默认夹爪采集的，而你当前使用的是“吸盘 + 六维力传感器”。这意味着原始 demo 的抓取/释放接触语义不能直接拿来完整重放。拆成两阶段的好处是：

1. 第一阶段先忠实记录“自动前段 + 人工后段”的真实执行过程。
2. 浏览器会先播放一遍原始成功轨迹，帮助用户确认目标物体和目标位置，减少误操作。
3. 第二阶段再统一做官方风格的回放、`no-op` 清洗和成功筛选。
4. 这样既保留了官方 regenerate 脚本的稳定结构，又适配了吸盘接触逻辑。

## 六维力字段说明

这个目录下的脚本统一使用：

- `obs/suction_wrench`

作为六维力/力矩观测字段。浏览器页面现在也会显示：

- `接触瞬时外力（世界系）`
- `总外载荷调试值（世界系）`

这样可以直接区分“实时接触变化”和“总载荷变化”。

## 推荐使用顺序

1. 先运行第一阶段原始采集脚本，得到“未清洗”的原始 HDF5。
2. 用本地电脑浏览器打开页面，先看原始轨迹预览，再决定是否开始采集。
3. 采集过程中根据页面说明使用按钮和键盘。
4. 再运行第二阶段清洗脚本，得到“删除 no-op 且删除失败轨迹”的最终 HDF5。

## 第一阶段运行示例

先激活环境：

```bash
conda activate openvla
```

然后运行，例如：

```bash
python experiments/robot/libero/data_regeneration/collect_libero_suction_handover_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial
```

注意：

- 现在不需要再手动传 `--libero_target_dir`
- 目标目录会自动固定到 `/data1/ps/openvla/data/{任务套件名}`

## 页面与交互说明

浏览器页面会实时显示：

- 当前阶段是什么
- 当前按钮分别有什么作用
- 当前模式是自动还是人工
- 六维力数值
- 当前附加状态信息

浏览器上的职责分工是：

- 页面按钮：
  - 负责 `播放原始LIBERO轨迹 / 数据采集 / 重播 / 退出`
- 键盘：
  - 负责人工采集时的机械臂控制
  - `H`：切人工
  - `R`：重启当前 demo
  - `SPACE`：切换吸盘开关
  - `Q / ESC`：退出脚本

## 额外说明

- 第一阶段脚本已经按高密度中文注释方式整理，目的是方便直接阅读代码理解每一步。
- 第二阶段脚本仍然保持“官方风格离线清洗”的定位，不需要浏览器参与。
- 如果你要在另一台机器上运行，建议先安装：
  - `pip install fastapi "uvicorn[standard]" pillow`
  - 如果你希望 JPEG 编码更快，可以额外安装 `opencv-python`
