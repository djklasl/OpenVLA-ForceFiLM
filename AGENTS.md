# Repository Guidelines

- 未经用户明确允许，不得运行任何指令和代码命令（包括py代码文件）。
- 未经用户明确允许，不得修改、创建、删除、重命名任何文件（包括py代码文件）。
- 创建新的代码文件，需要首先在文件的顶行位置进行中文注释，注释下这个代码文件是干什么的，输入是什
  么，输出是什么。然后再对代码行进行详细的逐行注释，要求小白也可以看懂。
- 修改代码文件的时候，也需要详细中文注释，要求小白也可以看懂。
- 不允许kill掉任何一个进程节点。

## 项目结构与模块组织
`prismatic/` 包含核心模型、训练流程和数据处理逻辑。`vla-scripts/` 提供 OpenVLA 微调与部署入口。`experiments/robot/` 存放 Bridge 和 LIBERO 评估脚本。`LIBERO/` 是内置基准测试代码与资源。`scripts/` 主要保留基础或历史工具。`experiments/logs/`、`rollouts/`、`checkpoint/` 为运行产物目录，提交前应确认未误加入无关文件。

## 协作与安全要求
未经仓库所有者明确允许，不得擅自新增、修改或删除本仓库中的任何文件内容。需要进行代码或文档变更时，先说明目标，再执行修改。与本仓库相关的终端命令一律先激活环境：`conda activate openvla`。与仓库所有者沟通时默认使用中文。

## 构建、测试与开发命令
先运行 `conda activate openvla`。

`pip install -e .`：以可编辑模式安装项目。  
`pip install -e .[dev]`：安装开发依赖，包括 `black`、`ruff` 和 `pre-commit`。  
`make check`：运行 `black --check` 和 `ruff check`，只检查不改文件。  
`make autoformat`：使用 `black` 与 `ruff --fix` 自动格式化。  
`make clean`：清理 `*.pyc` 和 `__pycache__`。  
示例：`python experiments/robot/libero/run_libero_eval.py --model_family openvla ...` 用于运行 LIBERO 评估。

## 代码风格与命名约定
本仓库以 Python 为主，使用 Black 和 Ruff 统一风格。行宽限制为 121，配置见 `pyproject.toml`。模块、函数、脚本文件统一使用 `snake_case`；新脚本名称应直接体现用途，例如 `run_libero_eval.py`。

## 测试说明
当前仓库没有统一的 `tests/` 目录，验证主要依赖静态检查和定向脚本测试。提交前至少运行 `make check`，并补充与你改动直接相关的脚本验证。现有零散测试文件包括 `LlamaV2/test_tokenizer.py` 和 `LIBERO/test_*.py`，可作为补充参考，但不能替代针对改动路径的验证。

## 提交与 Pull Request 规范
近期提交信息整体较短、直接，建议继续使用简洁祈使句风格，明确说明改动主题。Pull Request 应包含：改动内容、改动原因、执行过的验证命令；如果影响评估流程、日志或可视化结果，附上关键日志片段或截图。
