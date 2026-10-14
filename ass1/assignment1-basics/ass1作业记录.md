### 1. 终端命令

#### 1.1 数据准备

```shell
curl.exe -L --fail --retry 3 -o TinyStoriesV2-GPT4-train.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
curl.exe -L --fail --retry 3 -o TinyStoriesV2-GPT4-valid.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
curl.exe -L --fail --retry 3 -o owt_train.txt.gz "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
tar -xf owt_train.txt.gz
curl.exe -L --fail --retry 3 -o owt_valid.txt.gz "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
tar -xf owt_valid.txt.gz
```

#### 1.2 登录开发机

**内网或者先连接学校 VPN**

```
ssh 用户名@172.23.166.144 -p 37630 -i "私钥文件绝对路径"
```

登录后，主要文件应放在：

```
/ai/lmfs2026/用户名/
```

#### 1.3 激活环境

```shell
source /env/activate_a1.sh

# 验证
python --version
uv --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

首次准备时，复制课程项目：

```
cp -r /data/assignment1-basics \
  /ai/lmfs2026/$USER/assignment1-basics

cd /ai/lmfs2026/$USER/assignment1-basics
```

测试改用激活后的 Python：

```
python -m pytest --collect-only -q
python -m pytest
```

公共只读数据路径：

```
/data/assignment1-data/owt_train.txt
/data/assignment1-data/owt_valid.txt
```

#### 1.4 W&B 记录

首次登录：

```
wandb login
```

按提示输入自己的 API Key。然后配置：

```
export WANDB_ENTITY="你的实际Entity"
export WANDB_PROJECT="lmfs-assignment1"
export WANDB_MODE="online"
export PYTHONIOENCODING="utf-8"
export STUDENT_ID="你的学号"
```

注意，W&B 登录显示的用户名不一定就是可用的 Entity。网站提供了 `environment-test` 小实验；不确定时先按它验证上传，并记录实际打印的 Entity。

建议采用网站给出的记录与入口结构：

```
assignment1-basics/
    wandb_record.py          网站提供的公共实验记录工具
    bpe.py                  自己实现 BPE
    train_bpe.py            网站提供的 BPE 实验入口
    transformer_module.py  自己实现模型组件
    transformer_main.py    自己实现训练并接入课程入口
    tests/
        adapters.py
    outputs/               分词器、模型、运行产物
```

正式运行时需要：

- 启动时打印完整源码。
- 保存源码快照。
- BPE 与 Transformer 分别创建独立 run。
- 同一正式实验只初始化一个 run。
- 运行过程中不修改算法源码。



### 2. 开发流程

```
任务一训练并保存 BPE
        ↓
固定词表、merges、特殊 token 与 ID
        ↓
任务二加载同一份 BPE
        ↓
使用本人编码逻辑处理训练集和验证集
        ↓
训练 Transformer，并计算最终 PPL
```

#### 2.1 任务一

| 阶段 | 实现                                      | 涉及文件                                   | 远端验收                             |
| ---- | ----------------------------------------- | ------------------------------------------ | ------------------------------------ |
| B1   | 理解 bytes、预分词和特殊 token 边界       | 创建 `bpe.py`；必要时加一个小示例          | 小文本切分与手算结果一致             |
| B2   | 实现简单、容易理解的 BPE 训练             | 扩展 `bpe.py`；只接训练 adapter            | 合并顺序、频率和平局处理正确         |
| B3   | 实现 `encode`、`decode` 和特殊 token 识别 | 扩展 `bpe.py`；接 tokenizer adapter        | Unicode、空白、特殊 token 编解码正确 |
| B4   | 保存加载、惰性流式编码                    | 继续完善 `bpe.py`                          | 重载前后结果一致，流式边界正确       |
| B5   | 根据实际瓶颈优化                          | 只改已证明慢的部分                         | 正确性不变，实测性能改善             |
| B6   | 接入网站 W&B 记录器和 BPE 入口            | 逐步添加 `wandb_record.py`、`train_bpe.py` | 小规模流程能产生完整审查记录         |
| B7   | 服务器运行正式 OWT 实验                   | 使用已验证代码                             | 产生正式 BPE run 和分词器产物        |

BPE输入表示：

1. 用特殊 token 将语料切成互不连通的片段。

2. 对每个片段分别执行 GPT-2 正则预分词，使用 `regex.finditer`，空格通常会和后面的词一起成为一个 pre-token。

3. 将每个 pre-token 编码为 UTF-8。

4. 把 UTF-8 展开成单字节 `bytes` 元组，供下一步统计相邻 token 对。
