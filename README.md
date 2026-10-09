# M5 v2 / v3 — 课程提交代码

本仓库用于课程代码提交，覆盖 M5 的全部 10 家门店、30,490 条商品—门店序列。每店使用非递归和递归 LightGBM，以完整 12 层级 WRMSSE 评价和融合。这里的 **v2 是修正训练／验证重叠后的全量时间验证版本**，不是最早的冠军复现或 CA_1 原版脚本。论文另交 PDF，不放入这个仓库。

## 两个版本

|项目|v2|v3|
|---|---|---|
|正式训练轮数|两种模式均 3000|由前一历史窗口按模式全门店选择|
|融合|非递归50%／递归50%|前一历史窗口搜索统一融合权重|
|正式 CPU 线程|4|12，来自实际最终运行配置|
|轮数候选|不搜索|500、1000、1500、2000、2500、3000|
|递归权重候选|不搜索|0、0.1、…、1|
|已完成旧实验所选配置|固定3000/3000|开发500/500；Public及最终1000/500；均为非递归60%／递归40%|

v3 保留搜索步骤，不硬编码上述历史选择结果。先为两个单模型分别选择全门店 WRMSSE 最低的轮数，再固定轮数选择权重；不做轮数×轮数×权重联合搜索。精确同分取候选表第一个最小值：升序轮数下优先较小轮数，递归占比0→1下优先较小递归占比。没有近似同分容差。模型采用学习率0.015、Tweedie power 1.1等原参数；详细配置保存在两个版本的 `config.json`。

|阶段|正式训练历史|正式预测窗口|v3 选参历史截止／标签窗口|
|---|---|---|---|
|开发 development|d1–1885|d1886–1913|截至1857／d1858–1885|
|Public test|d1–1913|d1914–1941|截至1885／d1886–1913|
|最终 final|d1–1941|d1942–1969|继承 test 的选择，不使用 Public 或 Private 标签重新选择|

上市 release 周之前的行过滤方式保持原代码；历史起点为 d1，没有 FIRST_DAY=710 截断。目标编码按截止重算，只用此前全门店销量；递归先遮住28天未来销量，再逐日预测回填。日历和价格采用比赛提供的已知未来协变量设定：日历最多至截止+28，价格限于与该范围相交的周，边界周保留整周记录。

## 来源、引用与许可

代码来源：[Projectyctcmh 的 experiment/two-machine-v3 分支](https://github.com/Annnnnna11/Projectyctcmh/tree/experiment/two-machine-v3)。本机只读核对了该分支及此前已运行的 v2，源码对应记录如下：

- v2 训练记录 commit：`c0004a4a542c0deb0ece13b438ce92f6fe7d4147`。
- v3 训练记录 commit：`ec78fdb8cbc316c0ab2842d91898f399946477c1`；指定分支的结果归档 HEAD：`63621069e50ab2ee43f97fd63bcaeea55e5e4fca`。
- 逐文件来源、保留／排除清单及路径调整：[整理说明](docs/SUBMISSION_NOTES.md)、[源码来源清单](docs/source_provenance.json)。新仓库不继承原 Git 历史。

项目参考 M5 获胜方案的按门店特征及建模思路，但不声称复现全部冠军模型。评分定义参考代码中注明的 [Mcompetitions/M5-methods](https://github.com/Mcompetitions/M5-methods) 与 [Nixtla/datasetsforecast](https://github.com/Nixtla/datasetsforecast)。课程报告应注明源码来源及 M5 数据来源；参考方案／定义与直接复制代码应分别描述。

指定源仓库的已跟踪文件中未发现 LICENSE。本整理没有擅自声明 MIT 或其他开源授权；第三方库各自适用其许可，数据受 Kaggle 竞赛条款约束。若今后公开再分发此代码，应先确认源代码许可或获得相应授权。本轮仅建立本地课程提交仓库。

## 数据

在 [Kaggle M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) 获取并解压数据，按竞赛要求接受条款。将以下 **五份原始 CSV** 放在仓库的 `data/`：

```text
data/
  calendar.csv
  sell_prices.csv
  sales_train_validation.csv
  sales_train_evaluation.csv
  sample_submission.csv
```

运行清单对这五份文件都计算哈希。即使训练主要使用 evaluation 销量，也不要漏掉 validation 和 sample 文件。`sales_train_evaluation.csv` 只有至 d1941 的真实销量；本地不能计算 d1942–1969 的误差。不得用最终预测当作真值。

## 环境配置

推荐 Ubuntu 24.04 / WSL2、Python 3.12。源码使用 POSIX 的 `fcntl`、`resource`，macOS 也具备这些接口，但本精简仓库没有在 Mac 完整运行验证。Windows 原生 Python 不在已验证范围；Windows 用户在 WSL 中执行下列命令。CPU LightGBM 在 Linux 需要 OpenMP 运行库（Ubuntu 对应 `libgomp1`）；Mac 如缺少 OpenMP，应在自己的环境配置 `libomp`。

从仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_code.py
```

`requirements.txt` 固定实际运行记录中的六个直接依赖；`requirements-lock.txt` 补充它们在本机记录的传递依赖。如需匹配记录中的全部这些包，可改用 `pip install -r requirements-lock.txt`。没有复制原虚拟环境。Git 已初始化并有独立首次提交；运行时需要可用的 Git，因为 manifest 会记录当前 commit。

两种版本都优先使用仓库 `.venv`，不存在时用当前 Python，也可用 `M5_PYTHON` 指定解释器。默认路径由源码位置推导，与当前工作目录无关。自定义配置可设置 `M5_CONFIG` 为 JSON 路径；配置更改必须使用新 experiment 名称，断点续跑会核对配置、源码、依赖与输入哈希。默认目录均相对仓库；不要把原仓库缓存或旧 manifest 复制进新实验。

## 完整运行命令

下列数据检查和完整运行命令是供接收者运行的说明，**整理时没有执行它们**。正式运行默认串行、每店每模式单独进程，需自行安排时间和内存。

先检查原始数据：

```bash
.venv/bin/python src/check_raw_inputs.py --data-dir data
```

v2 全流程（预处理／特征、训练、预测、融合、评价，最后生成 Kaggle 文件）：

```bash
.venv/bin/python -u src/time_validation_v2/run.py all
.venv/bin/python scripts/make_submission.py --version v2
```

v3 全流程（包含两次独立历史选参、正式重训、预测、融合和评价）：

```bash
.venv/bin/python -u src/time_validation/run.py all
.venv/bin/python scripts/make_submission.py --version v3
```

也可以按阶段依次运行，保持开发→Public→最终顺序：

```bash
# v2
.venv/bin/python src/time_validation_v2/run.py stage --stage development
.venv/bin/python src/time_validation_v2/run.py stage --stage test
.venv/bin/python src/time_validation_v2/run.py stage --stage final
.venv/bin/python scripts/make_submission.py --version v2

# v3
.venv/bin/python src/time_validation/run.py stage --stage development
.venv/bin/python src/time_validation/run.py stage --stage test
.venv/bin/python src/time_validation/run.py stage --stage final
.venv/bin/python scripts/make_submission.py --version v3
```

各阶段中间步骤的参数保持原接口，见 `run.py --help`。例如 v3 的 `train_candidates`、`select_rounds`、`select_weight` 接受 `--cutoff`；`model`／`evaluate` 接受 `--stage`。单独调用中间步骤需要对应实验已有 manifest、缓存和前置产物；通常使用 `all` 或 `stage` 入口。v2 的开发后配置冻结检查保持不变。

状态查看：

```bash
.venv/bin/python src/time_validation_v2/status.py
.venv/bin/python src/time_validation/status.py
```

长任务可在 tmux 中运行以上相同命令；本仓库不启动定时任务或后台服务。旧机器8→12线程迁移脚本依赖旧产物，不在本课程仓库中。v3 默认直接在配置中记录实际12线程，不依赖本机外部 `sitecustomize.py`。

## 评价、输出及 Kaggle 文件

v2 输出根目录为 `experiments/time_validation_v2_submission/`，阶段目录是 `development_d1885/`、`test_d1913/`、`final_d1941/`。v3 根目录为 `experiments/time_validation_v3_submission/`，包括 `selection/cutoff_1857/`、`selection/cutoff_1885/`、`features/`、`development/`、`test/`、`final/`。

阶段的 `results/` 中保存单模型／融合／两个简单基线预测，以及回测 `scores.csv`、12层级分数、误差明细和特征重要性。两个简单基线是最后一周重复四次、最近四周同星期均值。`REPORT.md` 汇总进度和分数。WRMSSE 采用预测起点前历史缩放、前28天销售金额权重及12层等权汇总；零尺度／零收入处理与原实现一致。Bias 在原 scores 表中是平均预测减平均真实销量，`bias_percent` 才是总销量相对偏差百分数。

已有阶段产物后可仅重新评价，不重新训练：

```bash
.venv/bin/python src/time_validation_v2/run.py evaluate --stage test
.venv/bin/python src/time_validation/run.py evaluate --stage test
```

`make_submission.py` 只拼接已有预测并校验，不训练、不选权、不调用 Kaggle API：

- `_validation`：训练至 d1913，对 d1914–1941 的真实样本外 Public 预测。
- `_evaluation`：训练至 d1941，对 d1942–1969 的最终预测。
- 按 sample_submission 的60,980行、F1–F28排序，保持预测小数；不使用占位零填充任何一段。对应日期及权重另存 `.semantics.json`。

默认生成对应版本最终阶段 `results/kaggle_ensemble.csv`；可用 `--output` 指定路径。需要提交时由用户自行提交。代码不提供 Private 真实销量，也不报告其本地准确度。

## 可选跨机模块与原基线保护

v3 的 `handoff.py`、`collaborate.py`、`two_machine_plan.json` 被接收产物验证路径引用，因此保留；课程推荐使用上述单机全门店流程，不需要任何历史交接包。协作计划是相对任务配置，没有旧个人目录；旧一次性截止时间已设为 null，避免课程运行被过期时间阻止。两机交接协议、数据合同和哈希验证保留。

原 `docs/baseline_CA_1_manifest.json` 仅用于保护早期 CA_1 归档产物，本仓库不携带这些产物，因此默认没有该保护清单时跳过此项并明确记录。若配置 `M5_BASELINE_MANIFEST`，指定文件必须存在，且逐项哈希严格核对；存在清单时不会容忍产物被修改。训练、预测、评价仍需要自己生成的实验 manifest 和 checkpoint，不会因这一整理而跳过它们。原运行期修改 docs 的日志现写入实验目录 `CHANGELOG.md`。

## 验证范围与不提供的文件

检查结果见 [CHECKS.md](docs/CHECKS.md)。`scripts/check_code.py` 只执行语法、导入、配置、帮助命令和合成数据测试；已去除原 v3 测试中的 LightGBM 训练型前缀等价测试。真实的选参／正式训练代码没有因此删除。

**静态及轻量检查通过，不等于本精简仓库已完成真实全量运行验证。** 本轮没有训练、处理全量 M5、生成新的实验预测或评分，也没有安装／修改原项目环境。

不提供：论文PDF、原始数据、模型权重、预测CSV、完整历史结果归档、日志、缓存、虚拟环境、编辑器配置、AI工作记录、旧机器迁移／交接文档。上述运行产物目录被 `.gitignore` 排除。本轮没有创建 GitHub 仓库、配置远程或推送。
