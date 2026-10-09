# 精简整理说明

本仓库保留修正时间验证后的 v2 及带历史选参的 v3。原仓库和其产物仅只读使用，未复制 Git 历史。

## 保留

- `src/1_preprocessing_by_store.py`：两版实际特征定义所加载的模块；不是将早期训练脚本冒称 v2。
- `src/time_validation/`：v3 原目录与入口，保留 config、common、features、model、evaluate、run、status、合成数据 tests 及实际引用的 handoff／collaborate／two_machine_plan。
- `src/time_validation_v2/`：v2 同名接口独立存放，保持原 stage 参数。
- `src/m5_shared/`：两版共同的 WRMSSE、滞后／编码输入逻辑、特征表拼接、递归回填、原子写入和校验辅助，只存一份。
- `src/check_raw_inputs.py`：原分块数据审计，增加 import-safe 主入口和路径参数。
- `scripts/make_submission.py`：将真实 test 与 final 融合预测按官方样例顺序拼接。
- `scripts/check_code.py`：不触发训练的可重复检查。
- 两份 requirements、README、.gitignore、来源清单及本次检查说明。

## 排除

- 早期 `src/2_train/`、`src/3_predict/`、CA_1 `check_predictions.py`：不是本项目已运行 v2/v3 的入口。
- 原 `docs/results*` 全部历史归档、原始CSV、大型预测、模型、logs、cache、.venv、编辑器及代理配置。
- 原 launch、两机 smoke、外部参考运行脚本、旧线程迁移脚本与交接说明：主流程无需调用；运行方式在 README 中提供。
- 论文PDF：课程论文单独提交。

逐个上游已跟踪文件的排除理由在 `source_provenance.json` 中。保留的 source 文件逐项记录来源版本和哈希；对重复计算进行结构检查后才提取共享部分。

## 调整及其边界

1. ROOT 仍由源码位置推导；v2 子进程不再硬依赖固定 `.venv/bin/python`，优先项目虚拟环境，其次当前解释器，可由 M5_PYTHON 指定。v2/v3 支持 M5_CONFIG 指定配置文件。
2. 原数据审计的个人 home 路径改为仓库 data 默认路径与 --data-dir、--output 参数。
3. 实验名加 submission，防止与旧产物混淆；v3 默认12线程是历史实际最终配置，不是新增调参。v2仍4线程。训练参数、日期、特征、候选网格、同分规则不变。
4. CA_1 原归档保护清单改为可选保留项：未提供旧基线时不需要它；明确提供清单或已有清单时仍严格校验。自己的运行 manifest、数据集 schema 和 checkpoint 校验没有被删。
5. v3 原 docs 更新路径移到实验输出 CHANGELOG.md；不复制旧 AI／运行工作记录。
6. 两机模块因实际依赖保留。移除过期日程仅涉及工作分配配置，不改变预测算法。共享源码和新配置均进入运行指纹。
7. 共享 WRMSSE 的两版计算 AST 完全一致；共享特征和递归模块沿用原表达式。保留原函数接口或小包装，不重新设计模型。
8. 最早预处理模块在导入时不创建日志，首次实际日志事件才初始化 logger；计算函数未因此改变。
9. 删除原测试中的训练型前缀验证方法；不删除实际候选前缀预测／正式重训代码。本轮只运行剩余合成样本测试。

## 未解决／未验证

- 指定上游没有 LICENSE 文件；没有为上游代码擅自添加开源许可。未来公开再分发前需确认授权。
- 本轮未在新仓库完整训练；只验证语法、导入、路径、配置、文件依赖与小样本行为。大规模特征缓存未逐元素重建比较。
- 新仓库不会自动复用旧两机／8→12线程候选；保持同样算法重新运行，但不承诺逐位复现旧跨机混合来源的预测或精确 Kaggle 分数。
- Linux／WSL检查通过；Mac或Windows原生完整运行未验证，推荐Windows使用WSL。
