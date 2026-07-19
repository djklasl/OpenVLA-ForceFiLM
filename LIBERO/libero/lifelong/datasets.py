# 导入copy模块，用于对象的深拷贝（虽然本文件中未直接使用，但可能是预留的）
import copy

# 导入numpy库，用于高效的数值计算和数组操作
import numpy as np
# 导入robomimic框架的文件工具模块，用于从HDF5文件中读取数据集的元信息（如形状、维度等）
import robomimic.utils.file_utils as FileUtils
# 导入robomimic的观察值工具模块，用于初始化和处理观察数据的配置
import robomimic.utils.obs_utils as ObsUtils
# 从PIL库导入Image模块，用于图像处理（虽然本文件中未直接使用，但可能是预留的）
from PIL import Image
# 从robomimic导入SequenceDataset类，这是一个用于从HDF5文件中读取序列演示数据的基类
from robomimic.utils.dataset import SequenceDataset
# 从PyTorch导入Dataset基类，用于创建自定义数据集
from torch.utils.data import Dataset

"""
    这是一个从Robomimic框架借用的辅助函数，用于将HDF5格式的演示数据读取为序列数据集

    注意事项：robomimic的SequenceDataset有两个属性：seq_len和frame_stack
    原则上我们应该使用seq_len，但这两个属性的填充（padding）方式不同
    因此目前我们使用frame_stack而不是seq_len
"""


def get_dataset(
    dataset_path,           # HDF5数据集文件的路径
    obs_modality,           # 观察模态的配置，字典格式，例如：{"image": ["agentview_image"], "low_dim": ["robot_state"]}
    initialize_obs_utils=True,  # 是否初始化观察工具，默认为True
    seq_len=1,              # 序列长度，每个样本包含多少个连续的时间步，默认为1
    frame_stack=1,          # 帧堆叠数量，将相邻的多少帧堆叠在一起，默认为1
    filter_key=None,        # 过滤键，用于筛选特定类型的演示，默认为None（不过滤）
    hdf5_cache_mode="low_dim",  # HDF5缓存模式，"low_dim"表示缓存低维数据，"all"表示缓存所有数据
    *args,                  # 接收任意数量的位置参数（预留，未使用）
    **kwargs                # 接收任意数量的关键字参数（预留，未使用）
):

    # 如果需要初始化观察工具，则根据提供的观察模态配置初始化ObsUtils
    # 这一步是为了让robomimic知道如何处理不同类型的观察数据（图像、低维状态等）
    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    # 创建一个列表来存储所有的观察键（obs_keys）
    # 例如：如果obs_modality是{"image": ["front", "wrist"], "proprio": ["joint_pos"]}
    # 则all_obs_keys将变成["front", "wrist", "joint_pos"]
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():  # 遍历每种模态
        all_obs_keys += modality_list  # 将该模态下的所有观察键添加到列表中

    # 从HDF5数据集中获取形状元数据（shape metadata）
    # 这会返回一个字典，包含每个观察键的形状、类型等信息
    # verbose=False表示不打印详细信息
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    # 注意：这两行看起来是多余的（自我赋值），但可能是为了代码清晰或调试目的
    seq_len = seq_len
    filter_key = filter_key

    # 创建SequenceDataset对象，这是robomimic提供的数据集类
    # 它负责从HDF5文件中读取数据，并提供按序列访问的接口
    dataset = SequenceDataset(
        hdf5_path=dataset_path,           # HDF5文件的路径
        obs_keys=shape_meta["all_obs_keys"],  # 需要加载的观察数据键列表
        dataset_keys=["actions"],         # 需要加载的数据集键（这里是动作）
        load_next_obs=False,              # 不加载下一时刻的观察值
        frame_stack=frame_stack,          # 帧堆叠数量
        seq_length=seq_len,               # 序列长度（每个样本包含的时间步数）
        pad_frame_stack=True,             # 对帧堆叠进行填充（当序列长度不足时）
        pad_seq_length=True,              # 对序列长度进行填充（确保所有轨迹都能被采样）
        get_pad_mask=False,               # 不获取填充掩码
        goal_mode=None,                   # 不使用目标模式
        hdf5_cache_mode=hdf5_cache_mode,  # 缓存模式，将数据缓存在内存中以避免重复的文件I/O
        hdf5_use_swmr=False,              # 不使用SWMR（单写多读）模式
        hdf5_normalize_obs=None,          # 不对观察值进行归一化
        filter_by_attribute=filter_key,   # 根据属性过滤演示数据（可选）
    )
    # 返回创建的数据集对象和形状元数据
    return dataset, shape_meta

# 定义一个包装类SequenceVLDataset，继承自PyTorch的Dataset基类
# 这个类的用途是在序列数据的基础上添加任务嵌入（task embedding）
# VL可能代表Vision-Language，表示这是一个视觉-语言任务的数据集
class SequenceVLDataset(Dataset):
    # 初始化函数
    def __init__(self, sequence_dataset, task_emb):
        # 保存传入的序列数据集对象（SequenceDataset实例）
        self.sequence_dataset = sequence_dataset
        # 保存任务嵌入向量，这是一个用于标识任务的向量表示
        # 例如：可以用one-hot编码或语言模型的嵌入来表示不同的任务
        self.task_emb = task_emb
        # 从原始数据集中获取演示的数量（即有多少个演示轨迹）
        self.n_demos = self.sequence_dataset.n_demos
        # 从原始数据集中获取序列的总数量
        # 一个演示轨迹可以被分割成多个序列（取决于seq_length参数）
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    # 实现数据集的长度方法，返回数据集中的样本数量
    def __len__(self):
        # 直接返回原始序列数据集的长度
        return len(self.sequence_dataset)

    # 实现数据集的索引方法，根据索引获取一个样本
    def __getitem__(self, idx):
        # 从原始数据集中获取指定索引的样本
        # 返回的是一个字典，包含观察值、动作等数据
        return_dict = self.sequence_dataset.__getitem__(idx)
        # 将任务嵌入添加到返回的字典中
        # 这样每个样本都会附带其对应的任务信息
        return_dict["task_emb"] = self.task_emb
        # 返回增强后的字典
        return return_dict

# 定义一个多任务数据集类GroupedTaskDataset，用于处理多个任务的数据
# 这个类将多个任务的序列数据集组合在一起，并确保每个任务在训练中被平等地采样
# 这对于终身学习（Lifelong Learning）场景非常重要，可以避免某些任务的数据占主导地位
class GroupedTaskDataset(Dataset):
    # 初始化函数
    def __init__(self, sequence_datasets, task_embs):
        # 保存传入的序列数据集列表，每个元素是一个任务的SequenceDataset对象
        self.sequence_datasets = sequence_datasets
        # 保存每个任务对应的嵌入向量列表
        self.task_embs = task_embs
        # 任务组的数量（即有多少个不同的任务）
        self.group_size = len(sequence_datasets)
        # 计算所有任务的演示总数
        # 遍历每个数据集，累加它们的演示数量
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        # 计算所有任务的序列总数
        # 一个演示轨迹可以被分割成多个序列
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        # 获取每个数据集的长度（序列数量），存储为列表
        # 例如：如果有3个任务，分别有100、150、120个序列，则self.lengths = [100, 150, 120]
        self.lengths = [len(x) for x in self.sequence_datasets]
        # 任务组的大小，与group_size相同（冗余存储，可能用于某些特定场景）
        self.task_group_size = len(self.sequence_datasets)

        # 创建一个映射字典，将数据加载器的当前索引映射到原始任务数据的索引
        #
        # 核心思想：通过轮询（round-robin）方式组织数据，确保每个任务在每个轮次中都有代表
        #
        # 示例：假设我们有3个任务，分别有3、5、4个样本：
        # task-1  task-2  task-3
        #   0       1       2      <- 第1轮：每个任务取第0个样本
        #   3       4       5      <- 第2轮：每个任务取第1个样本
        #   6       7       8      <- 第3轮：每个任务取第2个样本
        #           9       10     <- 第4轮：task-1已无数据，取task-2和task-3的第3个样本
        #           11             <- 第5轮：只有task-2还有数据，取其第4个样本
        #
        # 这样做的好处是：当按顺序遍历数据集时，每个任务在每个轮次中都会被平等地采样
        # 避免了某些任务的数据因为数量多而主导训练过程

        # 初始化映射字典
        self.map_dict = {}
        # 将长度列表转换为numpy数组，方便操作
        sizes = np.array(self.lengths)  # 例如：[3, 5, 4]
        # row表示在某个任务数据集中的索引（行号）
        row = 0
        # col表示当前正在处理哪个任务（列号/任务ID）
        col = 0

        # 遍历所有样本的总数（sum(sizes)）
        for i in range(sum(sizes)):
            # 如果当前任务已经没有数据了（sizes[col] == 0），则跳到下一个任务
            while sizes[col] == 0:
                col = col + 1  # 移动到下一个任务
                # 如果超出任务数量，则回到第一个任务，并增加row
                if col >= self.task_group_size:
                    col -= self.task_group_size  # 回到第一个任务
                    row += 1  # 增加行号（表示该任务的下一个样本）

            # 将当前索引i映射到(row, col)
            # row表示在任务col中的样本索引，col表示任务ID
            self.map_dict[i] = (row, col)

            # 减少当前任务的剩余样本数（表示已经取了一个样本）
            sizes[col] -= 1
            # 移动到下一个任务
            col += 1
            # 如果超出任务数量，则回到第一个任务，并增加row
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1

        # 计算并保存数据集的总长度
        self.n_total = sum(self.lengths)

    # 实现数据集的长度方法
    def __len__(self):
        # 返回数据集的总长度（所有任务的样本总数）
        return self.n_total

    # 私有方法：根据合并后的索引获取原始任务的信息
    def __get_original_task_idx(self, idx):
        # 从映射字典中获取原始的索引信息
        # 返回一个元组：(样本在原任务中的索引, 任务ID)
        return self.map_dict[idx]

    # 实现数据集的索引方法
    def __getitem__(self, idx):
        # 获取原始索引信息：oi=原始样本索引, oti=原始任务ID
        oi, oti = self.__get_original_task_idx(idx)
        # 从对应的任务数据集中获取指定索引的样本
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        # 将对应任务的嵌入向量添加到返回字典中
        return_dict["task_emb"] = self.task_embs[oti]
        # 返回增强后的字典
        return return_dict

# 定义一个截断序列数据集类TruncatedSequenceDataset
# 这个类用于限制数据集的大小，只返回前buffer_size个样本
# 主要用于终身学习中的经验回放（Experience Replay）场景
# 当内存有限时，只保留最近或最重要的buffer_size个样本
class TruncatedSequenceDataset(Dataset):
    # 初始化函数
    def __init__(self, sequence_dataset, buffer_size):
        # 保存原始的序列数据集对象
        self.sequence_dataset = sequence_dataset
        # 设置缓冲区大小，即允许访问的最大样本数量
        # 例如：如果原始数据集有1000个样本，但buffer_size=100，
        # 则这个数据集只会返回前100个样本
        self.buffer_size = buffer_size

    # 实现数据集的长度方法
    def __len__(self):
        # 返回缓冲区的大小，而不是原始数据集的长度
        # 这限制了一次epoch中可以访问的样本数量
        return self.buffer_size

    # 实现数据集的索引方法
    def __getitem__(self, idx):
        # 直接从原始数据集中获取指定索引的样本
        # 注意：idx必须小于buffer_size，否则会越界
        return self.sequence_dataset.__getitem__(idx)
