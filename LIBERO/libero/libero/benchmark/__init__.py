# abc: Python的抽象基类模块，用于定义抽象接口
import abc
# os: 操作系统接口模块，用于文件路径操作
import os
# glob: 文件路径匹配模块（本文件中未直接使用，可能是预留的）
import glob
# random: 随机数生成模块（本文件中未直接使用，可能是预留的）
import random
# torch: PyTorch深度学习框架，用于加载初始状态文件
import torch

# typing模块的类型提示，用于静态类型检查
from typing import List, NamedTuple, Type
# get_libero_path: LIBERO项目的路径工具函数，用于获取数据集/配置文件的根目录
from libero.libero import get_libero_path
# libero_task_map: 从子模块导入的任务映射字典，包含各个benchmark的所有任务名称
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

# ============================================
# 全局变量：Benchmark注册表
# ============================================
# BENCHMARK_MAPPING: 一个全局字典，用于存储所有已注册的Benchmark类
# 键是benchmark类名的小写形式，值是对应的Benchmark类本身（不是实例）
# 这是一个典型的"注册器模式"实现
BENCHMARK_MAPPING = {}


# ============================================
# 装饰器函数：注册Benchmark类
# ============================================
def register_benchmark(target_class):
    """
    Benchmark注册器函数（通常作为装饰器使用）

    用法：
        @register_benchmark
        class MyBenchmark(Benchmark):
            ...

    作用：
        将传入的Benchmark类注册到全局字典BENCHMARK_MAPPING中
        类名会被转换为小写，使得查找时不区分大小写

    参数：
        target_class: 要注册的Benchmark类（是一个类对象，不是实例）
    """
    # target_class.__name__: 获取类的名字（字符串形式），如"LIBERO_SPATIAL"
    # .lower(): 转换为小写，实现大小写不敏感
    # 效果：BENCHMARK_MAPPING["libero_spatial"] = LIBERO_SPATIAL类
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


# ============================================
# 辅助函数：获取Benchmark注册表
# ============================================
def get_benchmark_dict(help=False):
    """
    获取所有已注册的benchmark字典

    参数：
        help: 如果为True，会打印出所有可用的benchmark名称

    返回：
        BENCHMARK_MAPPING字典，包含所有已注册的benchmark
    """
    # 如果请求帮助信息，打印所有可用的benchmark名称
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


# ============================================
# 核心函数：根据名称获取Benchmark类
# ============================================
def get_benchmark(benchmark_name):
    """
    根据benchmark名称获取对应的Benchmark类

    这是你notebook中导入的函数！

    参数：
        benchmark_name: benchmark的名称（字符串），如"libero_object"或"LIBERO_OBJECT"
                       大小写不敏感

    返回：
        对应的Benchmark类（不是实例，需要调用才能得到实例）

    示例：
        BenchmarkClass = get_benchmark("libero_object")
        benchmark = BenchmarkClass(task_order_index=0)
    """
    # .lower(): 实现大小写不敏感查找
    return BENCHMARK_MAPPING[benchmark_name.lower()]


# ============================================
# 调试函数：打印所有已注册的benchmark
# ============================================
def print_benchmark():
    """
    打印BENCHMARK_MAPPING字典的内容
    主要用于调试
    """
    print(BENCHMARK_MAPPING)


# ============================================
# 数据结构：Task任务描述
# ============================================
class Task(NamedTuple):
    """
    Task命名元组：描述一个任务的所有元信息

    NamedTuple是Python的一个轻量级数据类，类似于C语言的结构体
    相比普通元组，它可以通过字段名访问而不是索引

    字段说明：
        name: 任务名称（如"pick_up_the_cream_cheese_and_place_it_in_the_basket"）
        language: 任务的英文自然语言描述（用于生成语言嵌入）
        problem: 问题类型（通常是"Libero"）
        problem_folder: 任务所属的benchmark文件夹名（如"libero_object"）
        bddl_file: BDDL任务描述文件名（与name相同）
        init_states_file: 初始状态文件名（用于环境重置）

    使用示例：
        task = Task(
            name="pick_up_the_cream_cheese",
            language="pick up the cream cheese and place it in the basket",
            problem="Libero",
            problem_folder="libero_object",
            bddl_file="pick_up_the_cream_cheese.bddl",
            init_states_file="pick_up_the_cream_cheese.pruned_init"
        )
        print(task.language)  # 输出: pick up the cream cheese and place it in the basket
    """
    name: str           # 任务名称
    language: str       # 任务的自然语言描述
    problem: str        # 问题类型
    problem_folder: str # 任务文件夹
    bddl_file: str      # BDDL文件名
    init_states_file: str # 初始状态文件名


# ============================================
# 辅助函数：从文件名提取语言描述
# ============================================
def grab_language_from_filename(x):
    """
    从BDDL文件名中提取出自然语言描述

    LIBERO的命名规则：文件名本身就是用下划线分隔的英文短语
    例如："pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl"
    提取后："pick up the cream cheese and place it in the basket"

    参数：
        x: BDDL文件名（不带路径）

    返回：
        提取出的自然语言字符串
    """
    # 判断是否是LIBERO-100数据集（文件名以大写字母开头）
    if x[0].isupper():  # LIBERO-100
        # LIBERO-100有特殊的命名格式，包含SCENE信息
        if "SCENE10" in x:  # 处理SCENE10的特殊情况（两位数）
            # 从"SCENE10"后面开始提取，跳过8个字符（"SCENE10"）
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:  # SCENE1-SCENE9（单位数）
            # 从"SCENE"后面开始提取，跳过7个字符（"SCENE"）
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:  # LIBERO-10, LIBERO-Spatial, LIBERO-Object等
        # 直接用下划线分隔，然后拼接成空格分隔的句子
        language = " ".join(x.split("_"))

    # 移除文件扩展名".bddl"
    en = language.find(".bddl")
    return language[:en]


# ============================================
# 全局配置：LIBERO Benchmark套件列表
# ============================================
# libero_suites: 定义所有可用的LIBERO benchmark套件名称
# 每个套件包含一组不同类型的机器人操作任务
libero_suites = [
    "libero_spatial",  # 空间推理任务（如"把物体放到指定位置"）
    "libero_object",   # 物体操作任务（如"抓取特定物体并放置"）
    "libero_goal",     # 目标导向任务
    "libero_suction",  # 自定义吸盘任务套件
    "libero_90",       # 包含90个任务的大规模benchmark
    "libero_10",       # 包含10个任务的benchmark
]

# ============================================
# 全局变量：任务映射表
# ============================================
# task_maps: 一个嵌套字典，存储所有benchmark的所有任务信息
# 结构：task_maps[benchmark_name][task_name] = Task对象
# 例如：task_maps["libero_object"]["pick_up_the_cream_cheese..."] = Task(...)
task_maps = {}

# max_len: 最大语言描述长度（本变量未实际使用，可能是预留的）
max_len = 0

# ============================================
# 初始化代码：构建任务映射表
# ============================================
# 这段代码在模块导入时自动执行
# 遍历所有benchmark套件，为每个任务创建Task对象
for libero_suite in libero_suites:
    # 为当前benchmark创建一个空字典
    task_maps[libero_suite] = {}

    # 遍历当前benchmark中的所有任务
    # libero_task_map是在libero_suite_task_map.py中定义的字典
    for task in libero_task_map[libero_suite]:
        # 从文件名提取自然语言描述
        language = grab_language_from_filename(task + ".bddl")

        # 创建Task对象并存入映射表
        task_maps[libero_suite][task] = Task(
            name=task,                              # 任务名称
            language=language,                      # 自然语言描述
            problem="Libero",                       # 问题类型
            problem_folder=libero_suite,            # 所属文件夹
            bddl_file=f"{task}.bddl",               # BDDL文件名
            init_states_file=f"{task}.pruned_init", # 初始状态文件名
        )

        # 调试用的打印语句（已注释）
        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")


# ============================================
# 全局配置：任务顺序排列
# ============================================
# task_orders: 预定义的21种不同的任务学习顺序
# 每个列表代表一种任务排列顺序，用于终身学习实验
# 例如：[0,1,2,3,4,5,6,7,8,9]表示按原始顺序学习任务
#       [4,6,8,7,3,1,2,0,9,5]表示按打乱后的顺序学习
#
# 为什么需要多种顺序？
# - 终身学习研究需要测试算法在不同任务顺序下的表现
# - 有些顺序可能导致更严重的"遗忘"现象
# - 用于评估算法的鲁棒性和泛化能力
task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],      # 顺序0: 原始顺序
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],      # 顺序1: 打乱顺序1
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],      # 顺序2: 打乱顺序2
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],      # 顺序3: 打乱顺序3
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],      # 顺序4: 打乱顺序4
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],      # 顺序5: 打乱顺序5
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],      # 顺序6: 打乱顺序6
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],      # 顺序7: 打乱顺序7
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],      # 顺序8: 打乱顺序8
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],      # 顺序9: 打乱顺序9
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],      # 顺序10: 打乱顺序10
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],      # 顺序11: 打乱顺序11
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],      # 顺序12: 打乱顺序12
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],      # 顺序13: 打乱顺序13
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],      # 顺序14: 打乱顺序14
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],      # 顺序15: 打乱顺序15
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],      # 顺序16: 打乱顺序16
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],      # 顺序17: 打乱顺序17
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],      # 顺序18: 打乱顺序18
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],      # 顺序19: 打乱顺序19
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],      # 顺序20: 打乱顺序20
]


# ============================================
# 抽象基类：Benchmark基准测试
# ============================================
class Benchmark(abc.ABC):
    """
    Benchmark抽象基类

    这是所有LIBERO benchmark的基类，定义了通用的接口和行为
    使用abc.ABC使得这个类不能被直接实例化，必须被子类化

    子类示例：
        LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10, LIBERO_100
    """

    def __init__(self, task_order_index=0):
        """
        初始化Benchmark对象

        参数：
            task_order_index: 选择使用哪种任务顺序（0-20）
                             默认为0，表示使用原始顺序[0,1,2,...,9]
        """
        # task_embs: 任务嵌入向量列表，初始为None
        # 在训练前会通过set_task_embs()设置，用于表示每个任务的语义信息
        self.task_embs = None

        # task_order_index: 记录当前使用的任务顺序索引
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        """
        内部方法：构建benchmark的任务列表

        这个方法在子类初始化时被调用，负责：
        1. 从task_maps中获取所有任务
        2. 根据task_order_index选择任务顺序
        3. 设置self.tasks和self.n_tasks
        """
        # 获取当前benchmark的所有Task对象（按原始顺序）
        tasks = list(task_maps[self.name].values())

        # 特殊处理：libero_90 和 libero_suction 都使用固定任务顺序，不支持重排序
        if self.name in {"libero_90", "libero_suction"}:
            self.tasks = tasks
        else:
            # 根据task_order_index选择任务顺序
            # 打印信息，让用户知道当前使用的顺序
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            # 按照指定顺序重排任务列表
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]

        # 记录任务总数
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        """返回benchmark中的任务总数"""
        return self.n_tasks

    def get_task_names(self):
        """返回所有任务的名称列表"""
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        """返回所有任务的问题类型列表（通常都是["Libero", ...]）"""
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        """返回所有任务的BDDL文件名列表"""
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        """
        获取指定任务的BDDL文件的完整路径

        参数：
            i: 任务索引（0到n_tasks-1）

        返回：
            BDDL文件的绝对路径
        """
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),  # BDDL文件根目录
            self.tasks[i].problem_folder,   # benchmark文件夹
            self.tasks[i].bddl_file,        # BDDL文件名
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        """
        获取指定任务的演示数据集文件路径（相对路径）

        这个路径是相对于datasets文件夹的

        参数：
            i: 任务索引

        返回：
            演示数据集的相对路径，如"libero_object/pick_up_..._demo.hdf5"
        """
        # 检查索引是否在有效范围内
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"

        # 构造相对路径（相对于datasets文件夹）
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        """
        获取指定任务的Task对象

        参数：
            i: 任务索引

        返回：
            Task命名元组，包含任务的所有元信息
        """
        return self.tasks[i]

    def get_task_emb(self, i):
        """
        获取指定任务的任务嵌入向量

        参数：
            i: 任务索引

        返回：
            任务嵌入向量（通常是BERT或CLIP编码的向量）
        """
        return self.task_embs[i]

    def get_task_init_states(self, i):
        """
        获取指定任务的初始状态文件

        初始状态文件包含多个可用于重置环境的初始场景

        参数：
            i: 任务索引

        返回：
            初始状态张量
        """
        init_states_path = os.path.join(
            get_libero_path("init_states"),  # 初始状态文件根目录
            self.tasks[i].problem_folder,     # benchmark文件夹
            self.tasks[i].init_states_file,   # 初始状态文件名
        )
        # 使用torch加载预保存的初始状态
        init_states = torch.load(init_states_path)
        return init_states

    def set_task_embs(self, task_embs):
        """
        设置所有任务的任务嵌入向量

        这个方法通常在训练前调用，将生成的任务嵌入存入benchmark

        参数：
            task_embs: 任务嵌入向量列表，长度应等于n_tasks
        """
        self.task_embs = task_embs


# ============================================
# 具体Benchmark类：LIBERO_SPATIAL
# ============================================
@register_benchmark  # 装饰器：自动将此类注册到BENCHMARK_MAPPING
class LIBERO_SPATIAL(Benchmark):
    """
    LIBERO空间推理Benchmark

    包含10个侧重空间推理的机器人操作任务
    例如：将物体移动到指定位置、根据空间关系操作物体等
    """
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)  # 调用父类初始化
        self.name = "libero_spatial"                         # 设置benchmark名称
        self._make_benchmark()                               # 构建任务列表


# ============================================
# 具体Benchmark类：LIBERO_OBJECT
# ============================================
@register_benchmark
class LIBERO_OBJECT(Benchmark):
    """
    LIBERO物体操作Benchmark

    包含10个侧重物体操作的机器人操作任务
    例如：抓取特定物体、物体放置、物体操作等
    这是notebook示例中使用的benchmark！
    """
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


# ============================================
# 具体Benchmark类：LIBERO_GOAL
# ============================================
@register_benchmark
class LIBERO_GOAL(Benchmark):
    """
    LIBERO目标导向Benchmark

    包含10个目标导向的机器人操作任务
    任务具有明确的目标状态，需要机器人达成特定目标
    """
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


# ============================================
# 具体Benchmark类：LIBERO_SUCTION
# ============================================
@register_benchmark
class LIBERO_SUCTION(Benchmark):
    """
    LIBERO_SUCTION Benchmark

    这是一个自定义吸盘任务套件。
    任务集合来自人工筛选后的 23 个候选任务，顺序固定，不支持 task_orders 重排。
    """
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] libero_suction uses a fixed task order and does not support task_order_index != 0"
        self.name = "libero_suction"
        self._make_benchmark()


# ============================================
# 具体Benchmark类：LIBERO_90
# ============================================
@register_benchmark
class LIBERO_90(Benchmark):
    """
    LIBERO-90 Benchmark（大规模）

    包含90个任务的benchmark，用于大规模终身学习评估
    注意：由于任务数量多，不支持任务重排序功能
    """
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        # 断言：LIBERO_90只支持task_order_index=0（不支持重排序）
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


# ============================================
# 具体Benchmark类：LIBERO_10
# ============================================
@register_benchmark
class LIBERO_10(Benchmark):
    """
    LIBERO-10 Benchmark

    包含10个精选任务的benchmark
    常用于快速实验和原型验证
    """
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


# ============================================
# 具体Benchmark类：LIBERO_100
# ============================================
@register_benchmark
class LIBERO_100(Benchmark):
    """
    LIBERO-100 Benchmark（超大规模）

    包含100个任务的超大规模benchmark
    用于评估算法在大规模任务序列上的终身学习能力
    """
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()
