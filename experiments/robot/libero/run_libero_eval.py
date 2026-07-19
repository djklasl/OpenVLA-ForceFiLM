"""
run_libero_eval.py

【文件概述】
这个脚本用于在 LIBERO 仿真环境中运行和评估机器人学习模型。

【什么是LIBERO?】
LIBERO 是一个机器人操作任务的仿真基准测试平台，包含多种不同的任务场景，
比如空间推理(libero_spatial)、物体操作(libero_object)、目标达成(libero_goal)等。

【使用方法】
    # OpenVLA模型评估示例:
    # 重要提示: 如果模型是使用数据增强进行微调的，需要设置 `center_crop=True`

    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \                                    # 模型类型
        --pretrained_checkpoint <CHECKPOINT_PATH> \                 # 预训练模型检查点路径
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \  # 任务套件名称
        --center_crop [ True | False ] \                            # 是否使用中心裁剪
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \  # 可选的运行ID标签
        --use_wandb [ True | False ] \                              # 是否使用Weights & Biases进行日志记录
        --wandb_project <PROJECT> \                                 # W&B项目名称
        --wandb_entity <ENTITY>                                     # W&B实体名称
"""

# ==============================================================================
# 导入标准库
# ==============================================================================
import os              # 操作系统接口，用于文件和目录操作
import sys             # 系统相关功能，用于修改Python路径
from dataclasses import dataclass    # 数据类装饰器，用于简化配置类的定义
from pathlib import Path             # 路径处理库，提供面向对象的路径操作
from typing import Optional, Union   # 类型提示工具，用于标注参数和返回值类型

# ==============================================================================
# 设置使用的GPU（必须在导入torch之前设置）
# ==============================================================================
# 指定使用 cuda:2（即物理上的第3张GPU）
# 设置环境变量后，cuda:2 会被映射为程序中的 cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ==============================================================================
# 添加 LIBERO 库路径（必须在导入 libero 之前设置）
# ==============================================================================
# 将 LIBERO 目录添加到 Python 路径，使得可以导入 libero.libero
sys.path.insert(0, "/data1/ps/openvla/LIBERO")

# ==============================================================================
# 导入第三方库
# ==============================================================================
import draccus         # 配置管理库，用于解析命令行参数到配置类
import numpy as np     # 数值计算库，提供数组和矩阵操作
import tqdm            # 进度条库，用于显示循环进度

# 从libero库导入benchmark模块
# libero是一个机器人操作任务的仿真基准测试库
from libero.libero import benchmark

import wandb           # Weights & Biases，机器学习实验跟踪和可视化平台

# ==============================================================================
# 设置Python模块搜索路径
# ==============================================================================
# 将当前目录的上级目录添加到Python路径中
# 这样解释器就能找到 experiments.robot 模块
# 例如：如果当前在 /path/to/experiments/robot/libero/
# 这行代码会将 /path/to/experiments/ 添加到sys.path
sys.path.append("../..")

# ==============================================================================
# 导入项目内部模块
# ==============================================================================

# 从libero_utils导入LIBERO环境相关的工具函数
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,    # 获取空动作（用于等待物体稳定）
    get_libero_env,             # 创建和获取LIBERO仿真环境
    get_libero_image,           # 从观察数据中提取和处理图像
    quat2axisangle,             # 将四元数转换为轴角表示（旋转的另一种表示方式）
    save_rollout_video,         # 保存回合演示视频
)

# 从openvla_utils导入OpenVLA模型相关的处理工具
from experiments.robot.openvla_utils import get_processor

# 从robot_utils导入机器人相关的通用工具函数
from experiments.robot.robot_utils import (
    DATE_TIME,                  # 当前日期时间字符串，用于生成唯一的运行ID
    get_action,                 # 从模型获取动作
    get_image_resize_size,      # 获取图像调整后的尺寸
    get_model,                  # 加载和获取模型
    invert_gripper_action,      # 反转夹爪动作（开/关状态取反）
    normalize_gripper_action,   # 归一化夹爪动作到[-1, +1]范围
    set_seed_everywhere,        # 设置所有随机种子，确保实验可重复
)


# ==============================================================================
# 配置类定义
# ==============================================================================
@dataclass  # @dataclass装饰器会自动生成__init__、__repr__等方法
class GenerateConfig:
    """
    【配置类说明】
    这个类定义了评估脚本的所有可配置参数。
    使用dataclass装饰器可以让我们不用手写__init__方法，
    Python会自动根据类型注解生成初始化方法。

    使用方法：
        可以通过命令行参数覆盖这些默认值，例如：
        python run_libero_eval.py --model_family openvla --seed 42
    """

    # fmt: off  (这是格式化工具的指令，告诉代码格式化器不要调整下面的代码格式)

    #################################################################################################################
    # 模型相关参数
    #################################################################################################################
    model_family: str = "openvla"                    # 模型家族/类型，如 "openvla"
                                                     # 这决定了使用哪种模型架构

    pretrained_checkpoint: Union[str, Path] = ""     # 预训练检查点路径
                                                     # Union[str, Path] 表示可以是字符串或Path对象
                                                     # 这里存储模型权重文件的路径

    load_in_8bit: bool = False                       # (仅限OpenVLA) 是否使用8位量化加载模型
                                                     # 量化可以减少显存占用，但可能略微降低精度

    load_in_4bit: bool = False                       # (仅限OpenVLA) 是否使用4位量化加载模型
                                                     # 4位量化比8位更省显存，但精度损失更大

    center_crop: bool = True                         # 是否使用中心裁剪
                                                     # 如果训练时使用了随机裁剪数据增强，
                                                     # 评估时应该使用中心裁剪以保持一致性

    #################################################################################################################
    # LIBERO环境相关参数
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # 任务套件名称
                                                     # 可选值：
                                                     # - libero_spatial: 空间推理任务
                                                     # - libero_object: 物体操作任务
                                                     # - libero_goal: 目标达成任务
                                                     # - libero_10: 10个任务的综合测试
                                                     # - libero_90: 90个任务的综合测试

    num_steps_wait: int = 10                         # 等待步数
                                                     # 在仿真开始时，需要等待物体掉落并稳定
                                                     # 这段时间内不执行任何有效动作

    num_trials_per_task: int = 50                    # 每个任务的试验次数
                                                     # 即每个任务要运行多少个回合(episode)
                                                     # 次数越多，评估结果越准确

    #################################################################################################################
    # 工具参数
    #################################################################################################################
    run_id_note: Optional[str] = None                # 运行ID的附加备注
                                                     # Optional[str] 表示可以是字符串或None
                                                     # 用于在日志中标记不同的实验运行

    local_log_dir: str = "./experiments/logs"        # 本地日志目录
                                                     # 评估结果会保存到这个目录下的文本文件中

    use_wandb: bool = False                          # 是否使用Weights & Biases进行日志记录
                                                     # wandb是一个在线的机器学习实验跟踪平台

    wandb_project: str = "YOUR_WANDB_PROJECT"        # W&B项目名称
                                                     # 在wandb网站上创建的项目名

    wandb_entity: str = "YOUR_WANDB_ENTITY"          # W&B实体名称
                                                     # 通常是你的wandb用户名或团队名

    seed: int = 7                                    # 随机种子
                                                     # 设置相同的种子可以确保实验可重复
                                                     # 不同的种子会导致不同的随机初始化

    # fmt: on  (恢复格式化工具的正常行为)


# ==============================================================================
# 主评估函数
# ==============================================================================
@draccus.wrap()  # draccus装饰器，用于自动解析命令行参数并填充到cfg参数中
def eval_libero(cfg: GenerateConfig) -> None:
    """
    【函数说明】
    主评估函数，在LIBERO环境中评估模型的性能。

    【参数】
    cfg: GenerateConfig - 包含所有配置参数的对象

    【返回值】
    None - 这个函数不返回值，但会：
           1. 在控制台打印结果
           2. 将结果保存到本地日志文件
           3. (可选) 将结果上传到wandb
    """

    # ============================================================================
    # 参数验证
    # ============================================================================
    # 检查是否提供了模型检查点路径
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # 如果没有提供检查点路径，程序会报错并显示这条消息

    # 如果检查点路径中包含"image_aug"，说明模型训练时使用了图像增强
    # 那么评估时必须使用中心裁剪
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    # 不能同时使用8位和4位量化，这是互斥的选项
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # ============================================================================
    # 设置随机种子
    # ============================================================================
    # 设置随机种子以确保实验可重复
    # 这会影响numpy、random、torch等库的随机行为
    set_seed_everywhere(cfg.seed)

    # ============================================================================
    # 设置动作反归一化键
    # ============================================================================
    # [OpenVLA专用] 设置动作反归一化的键名
    # 模型输出的动作是归一化后的值，需要反归一化才能在环境中执行
    # 不同的任务套件有不同的归一化统计量
    cfg.unnorm_key = cfg.task_suite_name

    # ============================================================================
    # 加载模型
    # ============================================================================
    # 根据配置加载预训练模型
    # 这个函数会处理模型架构选择、权重加载、量化等
    model = get_model(cfg)

    # ============================================================================
    # 验证反归一化键是否存在
    # ============================================================================
    # [OpenVLA专用] 检查模型是否包含所需的反归一化统计量
    if cfg.model_family == "openvla":
        # 在某些情况下，数据集名称可能有后缀，需要手动调整键名
        # 例如，如果使用了"_no_noops"后缀的数据集版本
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

        # 确保反归一化键存在于模型的归一化统计量中
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # ============================================================================
    # 获取处理器
    # ============================================================================
    # [OpenVLA专用] 获取Hugging Face处理器
    # 处理器用于将图像和文本转换为模型可以理解的输入格式
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # ============================================================================
    # 初始化本地日志
    # ============================================================================
    # 生成唯一的运行ID，格式如: EVAL-libero_spatial-openvla-20240115_143022
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"

    # 如果有额外的备注，追加到运行ID后面
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # 创建日志目录（如果不存在）
    # exist_ok=True 表示如果目录已存在不会报错
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    # 构建日志文件的完整路径
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")

    # 打开日志文件，准备写入
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # ============================================================================
    # 初始化Weights & Biases日志
    # ============================================================================
    # 如果启用了wandb，初始化wandb运行
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,      # wandb实体（用户名或团队名）
            project=cfg.wandb_project,    # 项目名称
            name=run_id,                  # 本次运行的名称
        )

    # ============================================================================
    # 初始化LIBERO任务套件
    # ============================================================================
    # 获取所有可用的基准测试套件的字典
    benchmark_dict = benchmark.get_benchmark_dict()

    # 根据配置选择对应的任务套件
    # 例如 benchmark_dict["libero_spatial"]() 会返回libero_spatial任务套件的实例
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # 获取该任务套件中的任务数量
    num_tasks_in_suite = task_suite.n_tasks

    # 打印和记录任务套件信息
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # ============================================================================
    # 获取图像尺寸
    # ============================================================================
    # 获取模型期望的输入图像尺寸
    # 不同模型可能需要不同尺寸的图像输入
    resize_size = get_image_resize_size(cfg)

    # ============================================================================
    # 开始评估循环
    # ============================================================================
    # 初始化统计计数器
    total_episodes, total_successes = 0, 0  # 总回合数和总成功次数

    # 遍历任务套件中的每个任务
    # tqdm显示进度条，方便查看评估进度
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # ----------------------------------------------------------------
        # 获取当前任务
        # ----------------------------------------------------------------
        # 从任务套件中获取指定ID的任务对象
        task = task_suite.get_task(task_id)

        # 获取该任务的预设初始状态
        # 这些初始状态定义了任务开始时物体和机器人的位置
        initial_states = task_suite.get_task_init_states(task_id)

        # 创建LIBERO仿真环境
        # env: 仿真环境对象，用于执行动作和获取观察
        # task_description: 任务的文本描述，会作为模型的输入提示
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # ----------------------------------------------------------------
        # 对当前任务进行多次试验
        # ----------------------------------------------------------------
        # 初始化当前任务的统计计数器
        task_episodes, task_successes = 0, 0  # 当前任务的回合数和成功次数

        # 进行多次试验，每次试验是一个完整的回合
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            # 打印当前任务描述
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # ----------------------------------------------------------------
            # 重置环境
            # ----------------------------------------------------------------
            # 重置仿真环境到初始状态
            env.reset()

            # 设置特定的初始状态
            # 每个episode使用不同的预设初始状态
            obs = env.set_init_state(initial_states[episode_idx])

            # ----------------------------------------------------------------
            # 初始化回合变量
            # ----------------------------------------------------------------
            t = 0               # 时间步计数器
            replay_images = []  # 存储图像用于生成回放视频

            # 根据不同的任务套件设置最大步数
            # 这些值是根据训练数据中最长演示的步数设置的
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220   # 最长训练演示有193步
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280   # 最长训练演示有254步
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300   # 最长训练演示有270步
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520   # 最长训练演示有505步
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400   # 最长训练演示有373步

            # 打印回合开始信息
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            # ----------------------------------------------------------------
            # 主循环：执行动作直到任务完成或达到最大步数
            # ----------------------------------------------------------------
            # max_steps + cfg.num_steps_wait 是因为前面有等待步数不算在有效步数内
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # ----------------------------------------------------------------
                    # 等待物体稳定
                    # ----------------------------------------------------------------
                    # 重要：在前几个时间步不执行有效动作
                    # 因为仿真开始时物体会掉落，需要等待它们稳定下来
                    if t < cfg.num_steps_wait:
                        # 执行一个空动作（保持当前位置）
                        # get_libero_dummy_action 返回一个不改变机器人状态的动作
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1  # 增加时间步计数
                        continue  # 跳过本次循环的剩余部分，进入下一个时间步

                    # ----------------------------------------------------------------
                    # 获取和处理图像
                    # ----------------------------------------------------------------
                    # 从观察数据中提取图像并进行预处理
                    # 包括调整大小、裁剪等操作
                    img = get_libero_image(obs, resize_size)

                    # 保存处理后的图像，用于后续生成回放视频
                    replay_images.append(img)

                    # ----------------------------------------------------------------
                    # 准备观察数据字典
                    # ----------------------------------------------------------------
                    # 注意：OpenVLA模型不使用本体感知状态作为输入
                    # 但我们仍然准备它，以防其他模型需要
                    observation = {
                        "full_image": img,  # 相机图像
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],      # 机器人末端执行器的位置 (x, y, z)
                                quat2axisangle(obs["robot0_eef_quat"]),  # 末端执行器的朝向（四元数转轴角）
                                obs["robot0_gripper_qpos"]  # 夹爪的开合程度
                            )
                        ),
                    }
                    # state 拼接后的形状：[x, y, z, ax, ay, az, angle, gripper]
                    # 共7个值：3个位置 + 4个朝向(转换后为3个轴角) + 1个夹爪

                    # ----------------------------------------------------------------
                    # 查询模型获取动作
                    # ----------------------------------------------------------------
                    # 调用模型，根据当前观察和任务描述生成动作
                    # action 是一个7维向量：[dx, dy, dz, dax, day, daz, dgripper]
                    # 前6个是末端执行器的增量运动，最后1个是夹爪动作
                    action = get_action(
                        cfg,              # 配置对象
                        model,            # 模型对象
                        observation,      # 观察数据
                        task_description, # 任务描述（作为语言提示）
                        processor=processor,  # 图像/文本处理器
                    )

                    # ----------------------------------------------------------------
                    # 处理夹爪动作
                    # ----------------------------------------------------------------
                    # 将夹爪动作从 [0, 1] 归一化到 [-1, +1]
                    # binarize=True 会将连续值二值化为-1或+1
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA专用] 反转夹爪动作的符号
                    # 原因：OpenVLA的数据加载器将夹爪动作符号翻转以与其他数据集对齐
                    # (0 = 关闭, 1 = 打开)，所以需要翻转回来
                    # (-1 = 打开, +1 = 关闭) 以匹配环境的期望
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # ----------------------------------------------------------------
                    # 在环境中执行动作
                    # ----------------------------------------------------------------
                    # action.tolist() 将numpy数组转换为Python列表
                    # env.step 返回：
                    #   obs: 新的观察
                    #   reward: 奖励值（在这个任务中通常只关注done）
                    #   done: 任务是否完成（成功）
                    #   info: 额外信息字典
                    obs, reward, done, info = env.step(action.tolist())

                    # 如果任务完成（成功）
                    if done:
                        task_successes += 1      # 增加当前任务的成功计数
                        total_successes += 1     # 增加总成功计数
                        break  # 跳出while循环，结束当前回合

                    # 增加时间步计数
                    t += 1

                except Exception as e:
                    # 捕获并处理异常，防止程序崩溃
                    # 某些情况下可能会出现异常（如物理仿真不稳定）
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break  # 发生异常时跳出while循环

            # ----------------------------------------------------------------
            # 回合结束后的处理
            # ----------------------------------------------------------------
            # 更新计数器
            task_episodes += 1      # 当前任务完成的回合数
            total_episodes += 1     # 总完成的回合数

            # 保存回放视频
            # 这个视频记录了整个回合的执行过程
            save_rollout_video(
                replay_images,       # 图像序列
                total_episodes,      # 回合编号
                success=done,        # 是否成功
                task_description=task_description,  # 任务描述
                log_file=log_file    # 日志文件对象
            )

            # ----------------------------------------------------------------
            # 记录当前结果
            # ----------------------------------------------------------------
            # 打印到控制台
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # 写入日志文件
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()  # 刷新缓冲区，确保立即写入文件

        # ----------------------------------------------------------------
        # 当前任务所有试验完成后的总结
        # ----------------------------------------------------------------
        # 计算并打印当前任务的成功率
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # 写入日志文件
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

        # 如果启用了wandb，记录指标
        if cfg.use_wandb:
            wandb.log(
                {
                    # 记录当前任务的成功率
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    # 记录当前任务的回合数
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # ============================================================================
    # 评估结束，清理工作
    # ============================================================================
    # 关闭本地日志文件
    log_file.close()

    # 如果启用了wandb，上传最终结果
    if cfg.use_wandb:
        # 记录总体指标
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),  # 总成功率
                "num_episodes/total": total_episodes,  # 总回合数
            }
        )
        # 将本地日志文件也上传到wandb
        wandb.save(local_log_filepath)


# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    # 当脚本被直接运行时（而不是被导入时），执行评估函数
    # draccus.wrap() 装饰器会自动处理命令行参数
    eval_libero()
