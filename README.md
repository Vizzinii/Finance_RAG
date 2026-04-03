# Financial RAG Hallucination Evaluation System

面向金融场景的 RAG 问答与幻觉评测系统。项目将知识库管理、基于检索的问答、候选回答真实性评测、结果导出与可视化整合到一个 Streamlit Web 应用中，适合课程设计、毕业设计演示和论文答辩。

## 项目简介

本项目聚焦金融文本场景下的大模型幻觉检测问题，核心思路是：

- 使用本地知识库作为事实依据
- 对候选回答进行基于证据的核验
- 支持整体判定和 claim 级判定两种评测模式
- 提供结果导出、错误分析和 Prompt 模板管理

当前系统更偏向“演示优先、流程完整、结果可解释”的实现路线，适合围绕财报、政策报告、研报摘要等材料构建一个轻量级金融 RAG 评测平台。

## 核心功能

### 1. 知识库管理

- 支持上传 `PDF`、`TXT`、`DOCX`、`DOC`
- 自动完成文本读取、切分、向量化和本地持久化
- 使用 ChromaDB 作为本地向量数据库

### 2. RAG 问答

- 基于知识库进行检索增强问答
- 展示生成答案对应的证据片段
- 支持 OpenAI 兼容接口和 DashScope 兼容接口

### 3. 幻觉评测

- 支持导入结构化评测集
- 评测对象是 `question + candidate_answer`
- 支持两种模式：
  - `Overall Verdict`：整体判断回答是否被证据支持
  - `Claim-Level Verification`：先拆分 claim，再逐条核验
- 输出分类指标：
  - `Accuracy`
  - `Precision`
  - `Recall`
  - `F1`
  - `Uncertain Rate`

### 4. Prompt 模板管理

- 支持整体判定 Prompt
- 支持 Claim 拆解 Prompt
- 支持 Claim 核验 Prompt
- 支持保存、加载和恢复默认模板

### 5. 结果导出与错误分析

- 支持导出评测结果为 `JSON` 和 `CSV`
- 支持错误分析视图：
  - `Incorrect`
  - `Uncertain`
  - `False Positive`
  - `False Negative`

### 6. 配置文件驱动

- 项目启动时自动读取配置文件
- 模型 URL、模型名称、检索参数由配置文件统一管理
- 可在页面侧边栏保存和重新加载配置

## 技术栈

- Python 3.11
- Streamlit
- LangChain
- LangChain OpenAI
- ChromaDB
- OpenAI Compatible API / DashScope Compatible API
- Pandas
- Plotly
- pdfplumber
- python-docx
- pywin32

## 项目结构

```text
Finance_RAG/
├─ config/
│  └─ app_config.json              # 运行配置文件
├─ data/                           # 数据目录（向量库、示例数据等）
├─ src/
│  ├─ config_manager.py            # 配置管理
│  ├─ data_manager/
│  │  └─ test_set_manager.py       # 评测集管理
│  ├─ eval_engine/
│  │  ├─ hallucination_evaluator.py
│  │  ├─ prompt_manager.py
│  │  └─ result_exporter.py
│  ├─ knowledge_base/
│  │  ├─ document_loader.py
│  │  └─ vector_store_manager.py
│  ├─ rag_engine/
│  │  └─ financial_rag.py
│  └─ web_ui/
│     └─ app.py                    # Streamlit 入口
├─ reproduce_dashscope.py          # 独立调用示例
├─ test_kb.py
├─ test_rag_eval.py
├─ test_config_manager.py
├─ test_prompt_manager.py
├─ test_result_exporter.py
└─ README.md
```

## 安装与启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置项目

编辑 `config/app_config.json`。

当前配置结构如下：

```json
{
  "runtime": {
    "provider": "Aliyun DashScope (Qwen)",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "chat_model_name": "tongyi-xiaomi-analysis-pro",
    "embedding_model_name": "text-embedding-v1",
    "api_key": "YOUR_API_KEY",
    "vector_store_directory": "./data/chroma_db",
    "retrieval_top_k": 3
  },
  "provider_presets": {
    "OpenAI": {
      "base_url": "https://api.openai.com/v1",
      "chat_model_name": "gpt-3.5-turbo",
      "embedding_model_name": "text-embedding-ada-002"
    },
    "Aliyun DashScope (Qwen)": {
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "chat_model_name": "tongyi-xiaomi-analysis-pro",
      "embedding_model_name": "text-embedding-v1"
    },
    "Other": {
      "base_url": "http://localhost:8000/v1",
      "chat_model_name": "meta-llama/Llama-2-7b-chat-hf",
      "embedding_model_name": "text-embedding-ada-002"
    }
  }
}
```

注意：

- `api_key` 当前会明文保存在配置文件中，仅建议本地演示环境使用
- 如果仓库会推送到远程，请不要提交真实密钥

### 3. 启动应用

```bash
streamlit run src/web_ui/app.py
```

启动后默认访问：

- `http://localhost:8501`

## 使用流程

### 1. 配置模型

在左侧边栏中设置：

- `Provider`
- `Base URL`
- `Chat Model Name`
- `Embedding Model Name`
- `API Key`
- `Vector Store Directory`
- `Retrieval Top K`

修改后可点击：

- `Save Config`
- `Reload Config`

### 2. 导入知识库

在 `Data` 页上传知识库文档并点击 `Process and Add to KB`。

支持格式：

- `pdf`
- `txt`
- `docx`
- `doc`

说明：

- `docx` 直接解析
- `doc` 依赖 Windows 下的 Microsoft Word 自动化转换，建议本机已安装 Word

### 3. 导入评测集

在 `Data` 页点击 `Import Evaluation Dataset` 上传 JSON 文件，或者直接加载示例数据集。

### 4. 运行评测

在 `Eval` 页选择模式后点击 `Run Evaluation`：

- `Overall Verdict`
- `Claim-Level Verification`

### 5. 导出结果

评测完成后可直接下载：

- `evaluation_results.json`
- `evaluation_results.csv`

## 评测集格式

当前项目支持的评测集格式如下：

```json
{
  "dataset_name": "finance_eval_set",
  "kb_version": "",
  "retrieval_config": {},
  "samples": [
    {
      "id": 1,
      "question": "2024年末我国M2余额和同比增速分别是多少？",
      "candidate_answer": "2024年末，我国广义货币供应量（M2）余额为313.5万亿元，同比增长7.3%",
      "label": "negative",
      "source_model": "manual_grounded",
      "source_type": "manual",
      "reference_docs": [
        "2024年末，广义货币供应量（M2）余额为313.5万亿元，同比增长7.3%。"
      ],
      "ground_truth": "2024年末我国M2余额为313.5万亿元，同比增长7.3%",
      "notes": "演示样本"
    }
  ]
}
```

字段说明：

- `question`：问题
- `candidate_answer`：待评测回答
- `label`：`positive` 或 `negative`
- `source_model`：回答来源模型，可为空
- `source_type`：回答来源类型，如 `manual`、`weak_model`
- `reference_docs`：证据片段数组
- `ground_truth`：标准事实答案
- `notes`：附加说明

标签含义：

- `negative`：无幻觉，回答可被知识库支持
- `positive`：有幻觉，回答与证据冲突或存在证据外扩展

## 推荐演示数据策略

如果项目目标是系统演示和论文答辩，建议优先采用：

- 原始文档：完整财报、政策报告、研报
- 演示知识库：从原始文档中摘录出的核心事实 `txt`
- 评测样本：围绕这些核心事实构造的正负样本

这样做的优点是：

- 检索更稳定
- 证据更集中
- 演示效果更清晰
- 更容易解释评测结论

## 测试命令

```bash
python test_kb.py
python test_rag_eval.py
python test_config_manager.py
python test_prompt_manager.py
python test_result_exporter.py
```


## 注意事项

- `doc` 文件解析依赖本地 Word 环境
- 当前项目更适合轻量演示型知识库，不建议直接将大量长文档混入同一个演示库

## License

MIT License
