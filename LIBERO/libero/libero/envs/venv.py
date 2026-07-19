# ============================================
# 导入依赖库
# ============================================
# cloudpickle: 用于序列化Python对象，支持pickle无法处理的复杂对象（如lambda函数）
import cloudpickle
# ctypes: Python的外部函数库，用于与C语言代码交互
import ctypes
# gym: OpenAI Gym接口标准，定义了RL环境的基本接口
import gym
# numpy: 数值计算库，用于高效的数组操作
import numpy as np
import warnings  # 警告处理模块
import time       # 时间相关操作

# abc: 抽象基类模块，用于定义抽象接口
from abc import ABC, abstractmethod
# OrderedDict: 有序字典，保持插入顺序
from collections import OrderedDict
# Array, Pipe, connection: 多进程通信工具
from multiprocessing import Array, Pipe, connection
# Process: 多进程类，用于创建子进程
from multiprocessing.context import Process
# typing: 类型提示，用于静态类型检查
from typing import Any, Callable, List, Optional, Tuple, Union


# ============================================
# 类型定义：环境步进的返回值类型
# ============================================
# gym旧版API的返回类型: (obs, rew, done, info)
gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# gym新版API的返回类型: (obs, rew, terminated, truncated, info)
gym_new_venv_step_type = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
# 设置警告只显示一次，避免重复警告
warnings.simplefilter("once", DeprecationWarning)

# ============================================
# NumPy类型到C类型映射表
# ============================================
# 用于在多进程共享内存中将numpy数组映射到C语言类型
# 这样不同进程可以共享同一块内存区域
_NP_TO_CT = {
    np.bool_: ctypes.c_bool,      # 布尔类型
    np.uint8: ctypes.c_uint8,     # 8位无符号整数
    np.uint16: ctypes.c_uint16,   # 16位无符号整数
    np.uint32: ctypes.c_uint32,   # 32位无符号整数
    np.uint64: ctypes.c_uint64,   # 64位无符号整数
    np.int8: ctypes.c_int8,       # 8位有符号整数
    np.int16: ctypes.c_int16,     # 16位有符号整数
    np.int32: ctypes.c_int32,     # 32位有符号整数
    np.int64: ctypes.c_int64,     # 64位有符号整数
    np.float32: ctypes.c_float,   # 32位浮点数
    np.float64: ctypes.c_double,   # 64位浮点数
}
_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


# ============================================
# 辅助函数：弃用警告
# ============================================
def deprecation(msg: str) -> None:
    """弃用警告包装器。

    当用户使用即将废弃的功能时，发出 DeprecationWarning 警告。
    stacklevel=2 确保警告指向调用者的代码位置，而不是这个函数本身。

    知识点：Python 的警告系统
    - warnings.warn() 用于发出警告，而不是抛出异常
    - DeprecationWarning 表示某个功能在未来版本中将被移除
    - stacklevel 控制警告消息显示的堆栈层级
    """
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


# ============================================
# CloudpickleWrapper 类
# ============================================
class CloudpickleWrapper(object):
    """用于 SubprocVectorEnv 的 cloudpickle 包装器。

    知识点：为什么需要 Cloudpickle？
    - 标准的 pickle 模块无法序列化某些对象（如 lambda 函数、嵌套函数等）
    - cloudpickle 是一个更强大的序列化库，可以序列化几乎任何 Python 对象
    - 在多进程环境中，需要将环境函数序列化后传递给子进程

    使用场景：
    - 将环境创建函数 env_fn 传递给子进程时
    - 子进程需要反序列化并执行 env_fn() 来创建环境实例
    """

    def __init__(self, data: Any) -> None:
        # 保存需要序列化的数据（通常是环境创建函数）
        self.data = data

    def __getstate__(self) -> str:
        """当对象被 pickle 时调用，返回序列化后的字节流。

        知识点：Python 的 __getstate__ 和 __setstate__
        - __getstate__: 返回对象的状态字典，用于 pickle
        - __setstate__: 从状态字典恢复对象，用于 unpickle
        - 这里我们用 cloudpickle.dumps() 替代默认的 pickle
        """
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        """当对象被 unpickle 时调用，从字节流恢复对象。

        知识点：对象序列化流程
        1. 调用 pickle.dumps(obj) 时，会调用 obj.__getstate__()
        2. 调用 pickle.loads(data) 时，会调用 obj.__setstate__(data)
        3. 这里用 cloudpickle.loads() 替代默认的 pickle
        """
        self.data = cloudpickle.loads(data)


# ============================================
# Gym 环境保留键名常量
# ============================================
# 这些是 gym.Env 接口保留的属性名，不能被用作其他用途
# 当通过 BaseVectorEnv 访问这些属性时，会自动转发到底层环境
GYM_RESERVED_KEYS = [
    "metadata",          # 元数据（如渲染帧率等）
    "reward_range",      # 奖励范围（最小值、最大值）
    "spec",              # 环境规范（包含环境名称、版本等）
    "action_space",      # 动作空间定义（离散/连续，范围等）
    "observation_space", # 观测空间定义（图像形状、状态维度等）
]


################################################################################
#
# Worker 类：环境工作器
#
# 知识点：Worker 模式
# - Worker 是一个独立运行的环境实例，负责执行环境的操作
# - 每个 Worker 管理一个环境，可以串行或并行运行
# - 通过 send/recv 模式与主进程通信（异步模式）
# - 或者直接调用 step 方法（同步模式）
################################################################################


class EnvWorker(ABC):
    """环境工作器的抽象基类。

    知识点：抽象基类 (ABC)
    - ABC (Abstract Base Class) 定义了子类必须实现的接口
    - @abstractmethod 装饰器标记的方法必须在子类中实现
    - 这种设计模式确保所有 Worker 遵循相同的接口

    Worker 的作用：
    - 封装单个环境的所有操作（reset、step、render、close 等）
    - 提供统一的接口，使得 VectorEnv 可以管理不同类型的 Worker
    - 支持同步和异步两种工作模式
    """

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        # env_fn: 环境创建函数，调用 env_fn() 返回一个 gym.Env 实例
        self._env_fn = env_fn

        # is_closed: 标记环境是否已关闭
        # 用于防止在环境关闭后继续操作
        self.is_closed = False

        # result: 存储最后一次操作的结果
        # 可以是 step 返回值 (obs, rew, done, info)
        # 也可以是 reset 返回值 obs
        self.result: Union[
            gym_old_venv_step_type,
            gym_new_venv_step_type,
            Tuple[np.ndarray, dict],
            np.ndarray,
        ]

        # self.action_space = self.get_env_attr("action_space")  # noqa: B009
        # is_reset: 标记最后一次操作是否是 reset
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        """获取环境的属性。

        知识点：环境属性
        - action_space: 动作空间的定义（形状、范围等）
        - observation_space: 观测空间的定义
        - reward_range: 奖励的取值范围
        """
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        """设置环境的属性。"""
        pass

    def send(self, action: Optional[np.ndarray]) -> None:
        """向底层 Worker 发送动作信号。

        知识点：send/recv 模式（异步通信模式）
        - send: 发送请求但不等待结果（非阻塞）
        - recv: 接收上次请求的结果（阻塞直到有结果）
        - 这种模式可以在等待某些环境时，让其他环境继续运行

        信号类型：
        - action is None: 发送 "reset" 信号
        - action is not None: 发送 "step" 信号

        知识点：异步 vs 同步
        - 同步：发送请求后立即等待结果（step() 方法）
        - 异步：发送请求后继续做其他事，稍后再获取结果（send + recv）
        """
        if hasattr(self, "send_action"):
            # 向后兼容：如果子类实现了 send_action，发出弃用警告
            deprecation(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        """从底层 Worker 接收结果。

        知识点：返回值类型
        - 如果上次 send 的是 reset 信号：返回单个观测 obs
        - 如果上次 send 的是 step 信号：
          - 旧版 Gym API: (obs, rew, done, info)
          - 新版 Gym API: (obs, rew, terminated, truncated, info)

        知识点：terminated vs truncated（新版 Gym API）
        - terminated: 任务成功完成（如到达目标）
        - truncated: 因超时或其他原因被截断
        """
        if hasattr(self, "get_result"):
            # 向后兼容：如果子类实现了 get_result，发出弃用警告
            deprecation(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if not self.is_reset:
                self.result = self.get_result()
        return self.result

    @abstractmethod
    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """重置环境状态，返回初始观测。

        知识点：环境重置
        - 将环境恢复到初始状态
        - 通常在每个 episode 开始时调用
        - 返回环境的初始观测
        """
        pass

    def step(
        self, action: np.ndarray
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """执行环境的一个时间步。

        知识点：同步 vs 异步仿真
        - 同步仿真：send 和 recv 必须配对调用，用户只需调用 step()
        - 异步仿真：可以分开调用 send 和 recv，实现更灵活的控制

        为什么要分开 send 和 recv？
        - 在异步模式下，可以先对所有环境 send
        - 然后等待其中一部分完成（不用等待全部）
        - 这样可以充分利用 CPU，提高采样效率
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: List["EnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """等待多个 Worker 中的若干个完成操作。

        知识点：异步等待机制
        - workers: Worker 列表
        - wait_num: 等待多少个 Worker 完成
        - timeout: 超时时间（秒）

        使用场景：
        - 有 10 个并行环境
        - 只想等待其中 5 个完成就继续
        - 这样可以避免被最慢的环境拖累
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        """设置随机种子。

        知识点：随机种子的作用
        - 设置随机种子可以保证实验可复现
        - 每个环境应该有不同的种子，避免产生相同的轨迹
        - 通常用 seed + env_id 作为每个环境的种子
        """
        # return self.action_space.seed(seed)  # issue 299
        pass

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """渲染环境的视觉输出。

        知识点：环境渲染
        - 用于可视化环境状态
        - 可以是图像（RGB array）或打开窗口显示
        - 在训练时通常不开启渲染（太慢），只用于调试或评估
        """
        pass

    @abstractmethod
    def close_env(self) -> None:
        """关闭环境并释放资源。"""
        pass

    def close(self) -> None:
        """关闭 Worker。

        知识点：资源管理
        - is_closed 标记防止重复关闭
        - 重复关闭可能导致错误或资源泄漏
        """
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()


# ============================================
# ShArray 类：共享内存数组
# ============================================
class ShArray:
    """multiprocessing.Array 的包装器。

    知识点：进程间共享内存
    - 每个进程有独立的内存空间
    - 默认情况下，进程间不能直接共享数据
    - multiprocessing.Array 提供进程间共享的数组
    - 使用共享内存可以避免数据序列化的开销

    使用场景：
    - 在 SubprocEnvWorker 中，观测数据可以直接写入共享内存
    - 主进程可以直接从共享内存读取观测，无需通过管道传输
    - 对于大图像数据，这可以显著提高性能
    """

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        # dtype: NumPy 数据类型（如 np.float32）
        # shape: 数组形状（如 (128, 128, 3)）

        # _NP_TO_CT[dtype.type]: 将 NumPy 类型转换为 C 类型
        # int(np.prod(shape)): 计算数组元素总数
        # Array() 创建指定类型和长度的共享内存数组
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        """将 NumPy 数组保存到共享内存。

        知识点：零拷贝操作
        - get_obj() 获取共享内存的底层 C 数组
        - np.frombuffer() 将 C 数组包装为 NumPy 视图（不复制数据）
        - np.copyto() 将源数组复制到目标视图
        - 这样实现了数据从私有内存到共享内存的传输
        """
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()  # 获取底层共享内存的 C 数组
        # 将 C 数组包装为 NumPy 数组视图
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(
            self.shape
        )  # type: ignore
        # 将数据复制到共享内存
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        """从共享内存读取 NumPy 数组。

        知识点：内存视图
        - get_obj() 获取共享内存的底层 C 数组
        - np.frombuffer() 创建一个 NumPy 视图，不复制数据
        - 任何修改都会直接影响共享内存
        - 返回的数组只是原始数据的"窗口"
        """
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)  # type: ignore


# ============================================
# 辅助函数：为观测空间创建共享内存缓冲区
# ============================================
def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    """为给定的观测空间创建共享内存缓冲区。

    知识点：Gym 空间类型
    - Box: 连续值空间（如图像、关节角度）
    - Discrete: 离散动作空间（如 左/右/上/下）
    - Dict: 字典空间，包含多个命名的子空间
    - Tuple: 元组空间，包含多个子空间

    递归创建策略：
    - Dict 空间：为每个子空间递归创建，返回字典
    - Tuple 空间：为每个子空间递归创建，返回元组
    - Box 等其他空间：直接创建 ShArray

    使用场景：
    - 在子进程 Worker 初始化时，根据观测空间创建共享内存
    - 子进程将观测写入共享内存，主进程直接读取
    - 避免通过管道传输大量图像数据的开销
    """
    if isinstance(space, gym.spaces.Dict):
        # 字典空间：递归为每个子空间创建缓冲区
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        # 元组空间：递归为每个子空间创建缓冲区
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        # Box 等基本空间：直接创建共享数组
        return ShArray(space.dtype, space.shape)  # type: ignore


# ============================================
# Worker 子进程的主循环函数
# ============================================
def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    """子进程 Worker 的工作循环。

    知识点：多进程通信架构
    - 主进程 <-> 子进程 通过 Pipe（管道）通信
    - parent: 父进程端的管道连接（在子进程中被关闭）
    - p: 子进程端的管道连接
    - env_fn_wrapper: 包含环境创建函数的包装器
    - obs_bufs: 共享内存缓冲区（可选），用于高效传输观测

    命令协议：
    - 主进程发送 (cmd, data) 元组
    - cmd 是命令字符串，data 是参数
    - 子进程执行相应操作后，发送结果回来

    知识点：为什么要在子进程中运行环境？
    - Python 的全局解释器锁 (GIL) 限制了多线程的并行性
    - 多进程可以绕过 GIL，真正并行运行多个环境
    - 每个子进程有独立的 Python 解释器和内存空间
    """

    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        """将观测数据编码到共享内存缓冲区。

        知识点：递归编码
        - 如果观测是数组且缓冲区是 ShArray，直接保存
        - 如果观测是元组，递归编码每个元素
        - 如果观测是字典，递归编码每个值
        """
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    # 关闭父进程端的管道（只保留子进程端）
    # 知识点：管道是双向的，但每端应该只关闭自己不需要的一端
    parent.close()

    # 反序列化环境创建函数并创建环境实例
    env = env_fn_wrapper.data()

    try:
        # 主循环：持续接收命令并执行
        while True:
            try:
                # 接收命令：阻塞等待主进程发送命令
                cmd, data = p.recv()
            except EOFError:  # 管道已关闭（主进程结束）
                p.close()
                break

            # ========== STEP 命令 ==========
            if cmd == "step":
                # 执行环境步进
                env_return = env.step(data)

                # 如果有共享内存缓冲区，将观测写入共享内存
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    # 观测设为 None（主进程会从共享内存读取）
                    env_return = (None, *env_return[1:])
                p.send(env_return)

            # ========== RESET 命令 ==========
            elif cmd == "reset":
                # 执行环境重置
                retval = env.reset(**data)

                # 检查是否返回 info（新版 Gym API）
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval

                # 如果有共享内存缓冲区，将观测写入共享内存
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None

                # 发送结果
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)

            # ========== CLOSE 命令 ==========
            elif cmd == "close":
                # 关闭环境并退出循环
                p.send(env.close())
                p.close()
                break

            # ========== RENDER 命令 ==========
            elif cmd == "render":
                # 渲染环境
                p.send(env.render(**data) if hasattr(env, "render") else None)

            # ========== SEED 命令 ==========
            elif cmd == "seed":
                # 设置随机种子
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    # 如果环境没有 seed 方法，用 reset(seed=...)
                    env.reset(seed=data)
                    p.send(None)

            # ========== GETATTR 命令 ==========
            elif cmd == "getattr":
                # 获取环境属性
                p.send(getattr(env, data) if hasattr(env, data) else None)

            # ========== SETATTR 命令 ==========
            elif cmd == "setattr":
                # 设置环境属性
                # env.unwrapped 获取最原始的环境（去掉所有包装器）
                setattr(env.unwrapped, data["key"], data["value"])

            # ========== LIBERO 扩展命令 ==========
            elif cmd == "check_success":
                # LIBERO 特有：检查任务是否成功
                p.send(env.check_success())

            elif cmd == "get_segmentation_of_interest":
                # LIBERO 特有：获取分割掩码
                p.send(env.get_segmentation_of_interest(data))

            elif cmd == "get_sim_state":
                # LIBERO 特有：获取仿真器状态
                p.send(env.get_sim_state())

            elif cmd == "set_init_state":
                # LIBERO 特有：设置初始状态
                obs = env.set_init_state(data)
                p.send(obs)

            # ========== 未知命令 ==========
            else:
                p.close()
                raise NotImplementedError

    except KeyboardInterrupt:
        # 用户中断（Ctrl+C）
        p.close()


# ============================================
# DummyEnvWorker 类：串行环境工作器
# ============================================
class DummyEnvWorker(EnvWorker):
    """用于串行向量环境的虚拟 Worker。

    知识点：Dummy vs Subproc Worker
    - DummyEnvWorker: 在主进程中直接运行环境，无并行（串行）
    - SubprocEnvWorker: 在子进程中运行环境，真正的并行

    为什么叫 "Dummy"？
    - 它不是真正的并行 Worker，只是一个简单的包装器
    - 适用于环境数量较少、调试环境、或环境本身很快的场景
    - 优点：调试方便，无进程间通信开销
    - 缺点：无法利用多核 CPU，效率较低
    """

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        # 直接在主进程中创建环境实例
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        """直接获取环境属性（无需进程间通信）。"""
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        """直接设置环境属性（无需进程间通信）。

        知识点：env.unwrapped
        - env.unwrapped 获取最原始的环境对象
        - 如果环境被包装器（如 TimeLimit、RecordVideo）包装
        - unwrapped 会去掉所有包装器，返回原始环境
        """
        setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """直接重置环境（无需进程间通信）。"""
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        """等待 Worker 完成。

        知识点：串行 Worker 总是"就绪"的
        - 因为是串行执行，没有真正的异步等待
        - 所有的 Worker 立即返回，无需等待
        - wait_num 和 timeout 参数在这里没有实际作用
        """
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        """发送动作信号（立即执行）。

        知识点：Dummy Worker 的 send 是同步的
        - action is None: 执行 reset，保存结果到 self.result
        - action is not None: 执行 step，保存结果到 self.result
        - 与 SubprocEnvWorker 不同，这里立即执行，不发送到子进程
        """
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        """设置随机种子。"""
        super().seed(seed)
        try:
            return self.env.seed(seed)  # type: ignore
        except (AttributeError, NotImplementedError):
            # 如果环境没有 seed 方法，用 reset(seed=...)
            self.env.reset(seed=seed)
            return [seed]  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        """渲染环境。"""
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        """关闭环境。"""
        self.env.close()

    # ========== LIBERO 扩展方法 ==========

    def check_success(self):
        """LIBERO 特有：检查任务是否成功。"""
        return self.env.check_success()

    def get_segmentation_of_interest(self, segmentation_image):
        """LIBERO 特有：获取分割掩码。"""
        return self.env.get_segmentation_of_interest(segmentation_image)

    def get_sim_state(self):
        """LIBERO 特有：获取仿真器状态。"""
        return self.env.get_sim_state()

    def set_init_state(self, init_state):
        """LIBERO 特有：设置初始状态。"""
        return self.env.set_init_state(init_state)


# ============================================
# SubprocEnvWorker 类：子进程环境工作器
# ============================================
class SubprocEnvWorker(EnvWorker):
    """用于 SubprocVectorEnv 和 ShmemVectorEnv 的子进程 Worker。

    知识点：子进程 Worker 的优势
    - 每个环境在独立的子进程中运行
    - 绕过 Python 的 GIL（全局解释器锁）
    - 真正利用多核 CPU 并行运行多个环境
    - 适用于 CPU 密集型的仿真环境

    知识点：进程间通信 (IPC)
    - Pipe: 双向管道，用于父子进程通信
    - 共享内存: 用于高效传输大数组（如图像）
    """

    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        # 创建双向管道
        # parent_remote: 主进程端的管道连接
        # child_remote: 子进程端的管道连接
        self.parent_remote, self.child_remote = Pipe()

        # 是否使用共享内存传输观测
        # share_memory=True: 使用共享内存，减少数据复制
        # share_memory=False: 通过管道传输观测（较慢但更简单）
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None

        if self.share_memory:
            # 如果使用共享内存，需要先创建一个临时环境来获取观测空间
            # 知识点：为什么需要临时环境？
            # - 观测空间的形状和类型取决于环境
            # - 必须先创建环境才能知道观测空间的结构
            # - 创建后立即关闭，只用于获取空间信息
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            # 根据观测空间创建共享内存缓冲区
            self.buffer = _setup_buf(obs_space)

        # 准备子进程的参数
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),  # 序列化环境创建函数
            self.buffer,  # 共享内存缓冲区
        )

        # 创建并启动子进程
        # target=_worker: 子进程执行的主函数
        # args: 传递给 _worker 的参数
        # daemon=True: 设置为守护进程，主进程退出时自动结束
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()

        # 关闭子进程端的管道（主进程不需要）
        # 知识点：管道管理
        # - 每个端只需要保留自己需要的连接
        # - 关闭不用的端可以防止资源泄漏
        self.child_remote.close()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        """获取环境属性（通过进程间通信）。

        知识点：同步请求-响应模式
        - 1. send: 发送请求
        - 2. recv: 阻塞等待响应
        - 这是同步的，会等待子进程返回结果
        """
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        """设置环境属性（通过进程间通信）。"""
        self.parent_remote.send(["setattr", {"key": key, "value": value}])

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        """从共享内存解码观测数据。

        知识点：为什么要解码？
        - 子进程将观测写入共享内存
        - 主进程需要从共享内存读取
        - 递归处理字典、元组等嵌套结构
        """
        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]]
        ) -> Union[dict, tuple, np.ndarray]:
            if isinstance(buffer, ShArray):
                # 从共享数组读取数据
                return buffer.get()
            elif isinstance(buffer, tuple):
                # 递归解码元组
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                # 递归解码字典
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        """等待多个 Worker 中的若干个完成操作。

        知识点：异步等待机制
        - workers: 等待的 Worker 列表
        - wait_num: 需要等待多少个 Worker 完成
        - timeout: 超时时间（秒）

        知识点：connection.wait()
        - connection.wait() 是多路复用等待
        - 可以同时等待多个管道，返回就绪的管道
        - 这是实现异步并发的关键
        """
        # 获取所有 Worker 的管道连接
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []

        # 记录开始时间和剩余时间
        remain_time, t1 = timeout, time.time()

        # 循环直到有足够的 Worker 完成或超时
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                # 计算剩余时间
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break

            # 等待管道就绪
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore

            # 移除已就绪的连接，下次不再等待
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]

        # 返回就绪的 Worker 列表
        return [workers[conns.index(con)] for con in ready_conns]

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        """发送动作信号到子进程。

        知识点：非阻塞发送
        - send() 立即返回，不等待子进程完成
        - 这是异步模式的基础
        - 可以同时向多个环境发送动作
        """
        if action is None:
            # reset 信号
            if "seed" in kwargs:
                super().seed(kwargs["seed"])
            self.parent_remote.send(["reset", kwargs])
        else:
            # step 信号
            self.parent_remote.send(["step", action])

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        """从子进程接收结果。

        知识点：阻塞等待
        - recv() 会阻塞直到子进程发送结果
        - 如果使用共享内存，观测从共享内存读取
        """
        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            if len(result) == 2:
                # reset 返回值 (obs, info)
                obs, info = result
                if self.share_memory:
                    obs = self._decode_obs()
                return obs, info
            # step 返回值
            obs = result[0]
            if self.share_memory:
                obs = self._decode_obs()
            return (obs, *result[1:])  # type: ignore
        else:
            # 单个观测（旧版 reset）
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """重置环境（同步操作）。"""
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        self.parent_remote.send(["reset", kwargs])

        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            obs, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, info
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        """设置随机种子。"""
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        ret = self.parent_remote.recv()
        return ret

    def render(self, **kwargs: Any) -> Any:
        """渲染环境。"""
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        """关闭子进程和环境。

        知识点：进程关闭的正确流程
        1. 发送 close 命令给子进程
        2. 等待子进程响应（确保环境已关闭）
        3. join() 等待子进程退出
        4. terminate() 强制终止（兜底）

        为什么需要异常处理？
        - 如果子进程已经崩溃，send/recv 会失败
        - BrokenPipeError: 管道已断开
        - EOFError: 管道已关闭
        - AttributeError: multiprocessing 模块可能已卸载
        """
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()

    # ========== LIBERO 扩展方法 ==========

    def check_success(self):
        """LIBERO 特有：检查任务是否成功。"""
        self.parent_remote.send(["check_success", None])
        return self.parent_remote.recv()

    def get_segmentation_of_interest(self, segmentation_image):
        """LIBERO 特有：获取分割掩码。"""
        self.parent_remote.send(["get_segmentation_of_interest", segmentation_image])
        return self.parent_remote.recv()

    def get_sim_state(self):
        """LIBERO 特有：获取仿真器状态。"""
        self.parent_remote.send(["get_sim_state", None])
        return self.parent_remote.recv()

    def set_init_state(self, init_state):
        """LIBERO 特有：设置初始状态。"""
        self.parent_remote.send(["set_init_state", init_state])
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs


################################################################################
#
# VecEnvs: 向量化环境
#
# 知识点：向量化环境 (Vectorized Environments)
# - 同时运行多个相同的或不同的环境实例
# - 可以显著提高数据采集效率
# - 两大类：串行 (DummyVectorEnv) 和并行 (SubprocVectorEnv)
#
################################################################################


# ============================================
# BaseVectorEnv 类：向量化环境基类
# ============================================
class BaseVectorEnv(object):
    """向量化环境的基类。

    知识点：为什么需要向量化环境？
    - 单个环境的仿真速度有限，通常为 10-100 FPS
    - 训练强化学习模型需要大量样本
    - 并行运行多个环境可以成倍提高数据采集速度
    - 例如：8 个并行环境可以带来约 8 倍的加速

    使用示例：
    ::

        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    知识点：环境生成器 (env_fns)
    - env_fns 是一个环境创建函数的列表
    - env_fns[i]() 返回第 i 个环境实例
    - 使用函数而不是直接传入环境，是因为子进程需要独立创建环境

    基本用法：
    ::

        envs.seed(2)  # 为所有环境设置相同种子（递增）
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # 为每个环境设置不同种子
        obs = envs.reset()  # 重置所有环境
        obs = envs.reset([0, 5, 7])  # 只重置索引为 0, 5, 7 的环境
        obs, rew, done, info = envs.step([1] * 8)  # 同步步进所有环境
        envs.render()  # 渲染所有环境
        envs.close()  # 关闭所有环境

    知识点：异步模式 (wait_num 和 timeout)
    - wait_num: 等待多少个环境完成就返回
    - timeout: 最长等待时间（秒）
    - 适用于不同环境步进时间差异很大的场景
    - 例如：10 个环境，wait_num=5，只需等待 5 个完成即可

    :param env_fns: 环境创建函数列表，``env_fns[i]()`` 生成第 i 个环境
    :param worker_fn: Worker 创建函数，``worker_fn(env_fns[i])`` 生成第 i 个 Worker
    :param int wait_num: 异步模式下，等待多少个环境完成
    :param float timeout: 异步模式下的超时时间
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],  # 环境创建函数列表
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],  # Worker 创建函数
        wait_num: Optional[int] = None,  # 等待完成的环境数量
        timeout: Optional[float] = None,  # 超时时间（秒）
    ) -> None:
        # 保存环境创建函数列表
        self._env_fns = env_fns

        # 为每个环境创建一个 Worker
        # 知识点：Worker 池
        # - 一个 Worker 管理一个环境实例
        # - workers 列表保存所有 Worker 的引用
        # - 通过 Worker 统一管理不同的环境实现（串行/并行）
        self.workers = [worker_fn(fn) for fn in env_fns]

        # 获取 Worker 的类型（用于后续判断使用哪种 Worker）
        self.worker_class = type(self.workers[0])

        # 确保 Worker 是 EnvWorker 的子类
        # 知识点：类型检查
        # - issubclass 检查是否是子类关系
        # - isinstance 检查对象是否是某个类的实例
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        # 环境总数
        self.env_num = len(env_fns)

        # wait_num: 每次步进至少要等待多少个环境完成
        # 如果 wait_num 为 None，默认为 env_num（等待所有环境）
        # 知识点：wait_num < env_num 启用异步模式
        self.wait_num = wait_num or len(env_fns)

        # 检查 wait_num 是否在合法范围内
        assert (
            1 <= self.wait_num <= len(env_fns)
        ), f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"

        # timeout: 每次步进的最长等待时间
        # 如果设置了 timeout，也会启用异步模式
        self.timeout = timeout

        # 检查 timeout 是否为正数
        assert (
            self.timeout is None or self.timeout > 0
        ), f"timeout is {timeout}, it should be positive if provided!"

        # 判断是否启用异步模式
        # 异步模式条件：wait_num < env_num 或设置了 timeout
        self.is_async = self.wait_num != len(env_fns) or timeout is not None

        # 异步模式下的状态跟踪
        # waiting_conn: 正在等待的 Worker 连接列表
        self.waiting_conn: List[EnvWorker] = []

        # waiting_id: 已发送 action 但还未收到结果的 Worker 索引列表
        # 知识点：waiting_id vs ready_id
        # - waiting_id: 已发送 action 但还未收到结果的 Worker 索引
        # - ready_id: 当前可以接收 action 的 Worker 索引
        # 这两个列表在异步模式下动态更新
        self.waiting_id: List[int] = []

        # 初始状态下，所有环境都是就绪的
        # ready_id: 保存所有可以接收 action 的环境索引
        self.ready_id = list(range(self.env_num))

        # is_closed: 标记环境是否已关闭
        # 用于防止在环境关闭后继续操作
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        """确保环境未关闭。

        知识点：防御性编程
        - 在环境关闭后调用方法会导致错误
        - 提前检查并给出清晰的错误信息
        """
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """返回环境数量。"""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """根据属性类型切换获取方式。

        知识点：属性代理
        - 对于 Gym 保留属性（如 action_space），代理到 Worker
        - 对于其他属性，使用默认的获取方式
        - 这样可以让 BaseVectorEnv 表现得像一个 Gym 环境

        知识点：为什么需要代理？
        - VectorEnv 本身不是真正的 Gym 环境
        - 它只是一个管理多个环境的容器
        - 但为了让接口一致，需要代理 Gym 属性的访问
        """
        if key in GYM_RESERVED_KEYS:  # 检查是否是 Gym 保留属性
            return self.get_env_attr(key)  # 代理到 Worker 获取属性
        else:
            return super().__getattribute__(key)  # 使用默认方式获取属性

    def get_env_attr(
        self,
        key: str,  # 要获取的属性键名
        id: Optional[Union[int, List[int], np.ndarray]] = None,  # 环境索引
    ) -> List[Any]:
        """从底层环境获取属性。

        知识点：id 参数的灵活性
        - id=None: 获取所有环境的属性
        - id=0: 获取第 0 个环境的属性（返回单元素列表）
        - id=[0, 2, 4]: 获取指定环境的属性

        知识点：为什么返回列表？
        - 统一接口，无论获取多少个环境都返回列表
        - 方便批量操作
        """
        self._assert_is_not_closed()  # 确保环境未关闭
        id = self._wrap_id(id)  # 标准化 id 参数
        if self.is_async:  # 如果是异步模式
            self._assert_id(id)  # 检查环境是否可交互

        # 从每个 Worker 获取属性并返回列表
        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,  # 要设置的属性键名
        value: Any,  # 要设置的属性值
        id: Optional[Union[int, List[int], np.ndarray]] = None,  # 环境索引
    ) -> None:
        """设置底层环境的属性。"""
        self._assert_is_not_closed()  # 确保环境未关闭
        id = self._wrap_id(id)  # 标准化 id 参数
        if self.is_async:  # 如果是异步模式
            self._assert_id(id)  # 检查环境是否可交互
        # 遍历每个环境并设置属性
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,  # 输入的 id
    ) -> Union[List[int], np.ndarray]:
        """将 id 包装为统一格式。

        知识点：参数标准化
        - None -> [0, 1, 2, ..., env_num-1]
        - 0 -> [0]
        - [0, 1] -> [0, 1]
        """
        if id is None:  # 如果 id 为 None
            return list(range(self.env_num))  # 返回所有环境的索引
        return [id] if np.isscalar(id) else id  # 标量转为列表，非标量直接返回

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        """断言 id 对应的环境可以交互。

        知识点：异步模式下的状态检查
        - 不能与正在步进的环境交互（i not in waiting_id）
        - 只能与就绪的环境交互（i in ready_id）
        """
        for i in id:  # 遍历每个 id
            assert (
                i not in self.waiting_id
            ), f"Cannot interact with environment {i} which is stepping now."
            assert (
                i in self.ready_id
            ), f"Can only interact with ready environments {self.ready_id}."

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,  # 环境索引
        **kwargs: Any,  # 其他参数（如 seed）
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """重置部分或全部环境，返回初始观测。

        知识点：批量重置
        - id=None: 重置所有环境
        - id=[0, 5, 7]: 只重置指定环境
        - 返回堆叠后的观测数组 (env_num, *obs_shape)

        知识点：返回值类型
        - 旧版 Gym: 返回 obs
        - 新版 Gym: 返回 (obs, info)

        使用场景：
        - episode 开始时重置所有环境
        - 部分环境 done 后，只重置那些环境
        """
        self._assert_is_not_closed()  # 确保环境未关闭
        id = self._wrap_id(id)  # 标准化 id 参数
        if self.is_async:  # 如果是异步模式
            self._assert_id(id)  # 检查环境是否可交互

        # send(None) == reset() in worker
        # 知识点：send(None) 表示 reset 信号
        for i in id:  # 遍历每个要重置的环境
            self.workers[i].send(None, **kwargs)  # 发送 reset 信号

        # 接收所有环境的 reset 结果
        ret_list = [self.workers[i].recv() for i in id]

        # 检查 reset 是否返回 info（新版 Gym API）
        reset_returns_info = (
            isinstance(ret_list[0], (tuple, list))  # 是否是元组或列表
            and len(ret_list[0]) == 2  # 长度是否为 2
            and isinstance(ret_list[0][1], dict)  # 第二个元素是否是字典
        )

        if reset_returns_info:  # 如果返回 info
            obs_list = [r[0] for r in ret_list]  # 提取观测
        else:  # 如果只返回观测
            obs_list = ret_list

        # Tuple 观测空间不支持（需要改为 Array 或 Dict）
        if isinstance(obs_list[0], tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )

        # 知识点：np.stack vs np.array
        # np.stack: 沿新轴堆叠数组（要求形状相同）
        # np.array(..., dtype=object): 允许不同形状的数组
        try:
            obs = np.stack(obs_list)  # 尝试堆叠数组
        except ValueError:  # 如果形状不同
            obs = np.array(obs_list, dtype=object)  # 使用 object 类型

        if reset_returns_info:  # 如果返回 info
            infos = [r[1] for r in ret_list]  # 提取 info
            return obs, infos  # 返回观测和 info
        else:  # 如果只返回观测
            return obs  # 只返回观测

    def step(
        self,
        action: np.ndarray,  # 批量动作数组
        id: Optional[Union[int, List[int], np.ndarray]] = None,  # 环境索引
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """对部分或全部环境执行一步动力学模拟。

        知识点：同步 vs 异步步进
        - 同步模式 (is_async=False):
          - 等待所有指定环境完成步进
          - 简单直接，但可能被最慢的环境拖累
        - 异步模式 (is_async=True):
          - 只需等待 wait_num 个环境完成
          - 或者等待 timeout 时间
          - 更高效，适合环境速度差异大的场景

        知识点：返回值格式
        - 旧版 Gym API: (obs, rew, done, info)
        - 新版 Gym API: (obs, rew, terminated, truncated, info)

        知识点：info["env_id"]
        - 每个环境的 info 中都会包含 env_id
        - 用于区分是哪个环境返回的结果
        - 在异步模式下特别重要

        :param numpy.ndarray action: 代理提供的批量动作
        :param id: 要步进的环境索引（默认为全部）

        :return: 批量的观测、奖励、完成标志、信息
        """
        self._assert_is_not_closed()  # 确保环境未关闭
        id = self._wrap_id(id)  # 标准化 id 参数

        # ========== 同步模式 ==========
        if not self.is_async:  # 如果不是异步模式
            assert len(action) == len(id)  # 确保动作数量与环境数量一致
            for i, j in enumerate(id):  # 遍历每个环境
                self.workers[j].send(action[i])  # 发送动作

            result = []  # 存储结果
            for j in id:  # 遍历每个环境
                env_return = self.workers[j].recv()  # 接收结果
                env_return[-1]["env_id"] = j  # 添加 env_id 到 info 中
                result.append(env_return)  # 添加到结果列表

        # ========== 异步模式 ==========
        else:  # 如果是异步模式
            if action is not None:  # 如果有动作
                self._assert_id(id)  # 检查环境是否可交互
                assert len(action) == len(id)  # 确保动作数量与环境数量一致

                # 发送动作到所有指定的环境
                for act, env_id in zip(action, id):  # 遍历动作和环境索引
                    self.workers[env_id].send(act)  # 发送动作
                    self.waiting_conn.append(self.workers[env_id])  # 加入等待列表
                    self.waiting_id.append(env_id)  # 记录等待的环境索引

                # 从就绪列表中移除已发送动作的环境
                self.ready_id = [x for x in self.ready_id if x not in id]

            # 等待足够的环境完成
            ready_conns: List[EnvWorker] = []  # 已完成的 Worker 列表
            while not ready_conns:  # 循环直到有 Worker 完成
                ready_conns = self.worker_class.wait(  # 等待 Worker 完成
                    self.waiting_conn, self.wait_num, self.timeout
                )

            # 收集已完成的环境结果
            result = []  # 存储结果
            for conn in ready_conns:  # 遍历已完成的 Worker
                waiting_index = self.waiting_conn.index(conn)  # 获取索引
                self.waiting_conn.pop(waiting_index)  # 从等待列表移除
                env_id = self.waiting_id.pop(waiting_index)  # 获取环境索引

                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()  # 接收结果
                env_return[-1]["env_id"] = env_id  # 添加 env_id 到 info
                result.append(env_return)  # 添加到结果列表
                self.ready_id.append(env_id)  # 重新加入就绪列表

        # 堆叠结果为批量格式
        # 知识点：zip(*result) 转置结果列表
        # result = [(obs1, rew1, done1, info1), (obs2, rew2, done2, info2), ...]
        # zip(*result) = [(obs1, obs2, ...), (rew1, rew2, ...), ...]
        return_lists = tuple(zip(*result))  # 转置结果列表
        obs_list = return_lists[0]  # 获取观测列表

        # 尝试堆叠观测数组
        try:
            obs_stack = np.stack(obs_list)  # 堆叠数组
        except ValueError:  # 如果形状不同
            obs_stack = np.array(obs_list, dtype=object)  # 使用 object 类型

        # 堆叠其他返回值（rew, done, info 等）
        other_stacks = map(np.stack, return_lists[1:])
        return (obs_stack, *other_stacks)  # 返回堆叠后的结果

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,  # 种子值或种子列表
    ) -> List[Optional[List[int]]]:
        """为所有环境设置随机种子。

        知识点：种子策略
        - seed=None: 不设置种子（使用随机种子）
        - seed=2: 为环境设置递增种子 [2, 3, 4, 5, ...]
        - seed=[2, 3, 4, ...]: 为每个环境设置指定种子

        知识点：为什么每个环境需要不同种子？
        - 如果所有环境使用相同种子，它们的行为会完全一致
        - 这会导致数据多样性降低，影响训练效果
        - 使用递增种子可以保证每个环境的随机性

        :return: 每个环境使用的种子列表
        """
        self._assert_is_not_closed()  # 确保环境未关闭
        seed_list: Union[List[None], List[int]]  # 种子列表

        if seed is None:  # 如果没有指定种子
            seed_list = [seed] * self.env_num  # 所有环境使用 None
        elif isinstance(seed, int):  # 如果指定了一个整数
            # 递增种子：seed + i
            seed_list = [seed + i for i in range(self.env_num)]
        else:  # 如果指定了种子列表
            seed_list = seed  # 直接使用

        # 为每个 Worker 设置种子并返回结果
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """渲染所有环境。

        知识点：渲染的限制
        - 异步模式下，如果环境正在步进，无法渲染
        - 渲染通常只用于调试或评估
        - 训练时不开启渲染以提高速度
        """
        self._assert_is_not_closed()  # 确保环境未关闭

        # 如果是异步模式且有环境正在步进，抛出错误
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )

        # 渲染所有环境并返回结果列表
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """关闭所有环境。

        知识点：资源清理
        - 确保所有 Worker 正确关闭
        - 关闭子进程（如果有）
        - 释放共享内存（如果有）
        - 这个函数只应该被调用一次

        知识点：垃圾回收
        - 如果用户忘记调用 close()
        - Python 的垃圾回收机制会调用
        - 但最好显式调用以确保资源释放
        """
        self._assert_is_not_closed()  # 确保环境未关闭

        # 关闭所有 Worker
        for w in self.workers:
            w.close()

        # 标记环境已关闭
        self.is_closed = True


# ============================================
# DummyVectorEnv 类：串行向量化环境
# ============================================
class DummyVectorEnv(BaseVectorEnv):
    """串行实现的向量化环境包装器。

    知识点：Dummy vs Subproc
    - DummyVectorEnv: 使用 for-loop 串行执行
    - SubprocVectorEnv: 使用多进程并行执行

    使用场景：
    - 环境数量较少（如 1-4 个）
    - 调试代码（串行更容易调试）
    - 环境本身很快（串行开销不大）

    不适用场景：
    - 环境数量多（如 8+ 个）
    - 环境本身很慢（物理仿真）
    - 需要充分利用多核 CPU
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        # 使用 DummyEnvWorker 创建串行 Worker
        super().__init__(env_fns, DummyEnvWorker, **kwargs)

    # ========== LIBERO 扩展方法 ==========

    def check_success(self):
        """检查所有环境的任务是否成功。"""
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        """获取所有环境的分割掩码。"""
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        """获取所有环境的仿真器状态。"""
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """设置部分或全部环境的初始状态。

        知识点：初始状态设置
        - init_state: 要设置的初始状态列表
        - id: 要设置的环境索引
        - 返回设置后的观测
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # 逐个设置初始状态
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


# ============================================
# SubprocVectorEnv 类：并行向量化环境
# ============================================
class SubprocVectorEnv(BaseVectorEnv):
    """基于子进程的向量化环境包装器。

    知识点：多进程并行
    - 每个环境在独立的子进程中运行
    - 绕过 Python 的 GIL，真正并行
    - 适合 CPU 密集型的物理仿真环境

    使用场景：
    - 环境数量较多（如 8+ 个）
    - 环境本身很慢（物理仿真）
    - 需要充分利用多核 CPU

    优势：
    - 真正并行，效率高
    - 充分利用多核 CPU

    劣势：
    - 调试较困难（子进程）
    - 进程间通信有开销
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        # Worker 创建函数：使用 SubprocEnvWorker（不使用共享内存）
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)

    # ========== LIBERO 扩展方法 ==========

    def check_success(self):
        """检查所有环境的任务是否成功。"""
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        """获取所有环境的分割掩码。"""
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        """获取所有环境的仿真器状态。"""
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """设置部分或全部环境的初始状态。"""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # 逐个设置初始状态
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs
