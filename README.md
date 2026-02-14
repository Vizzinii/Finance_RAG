# Financial RAG & Hallucination Evaluation System (金融 RAG 与幻觉评测系统)

## 📖 项目简介 (Introduction)

这是一个面向金融场景的 RAG (检索增强生成) 系统，集成了知识库管理、智能问答以及大模型幻觉评测功能。系统旨在帮助金融分析师高效利用研报数据，并通过自动化的评测机制确保回答的准确性（信实度）和相关性。

本项目基于 Python 开发，使用 Streamlit 构建 Web 界面，底层采用 LangChain 框架结合 ChromaDB 向量数据库，支持 OpenAI (GPT) 和 阿里云通义千问 (DashScope/Qwen) 等多种大模型接口。

## ✨ 核心功能 (Key Features)

*   **📚 知识库管理 (Knowledge Base)**:
    *   支持上传 PDF 和 TXT 格式的金融研报。
    *   自动进行文本清洗、切片 (Chunking) 和向量化 (Embedding)。
    *   使用 ChromaDB 进行本地向量存储。
*   **💬 智能对话 (RAG Chat)**:
    *   基于知识库的金融问答。
    *   展示回答的参考来源 (Source Documents)。
    *   支持流式对话体验。
*   **⚖️ 幻觉评测 (Hallucination Evaluation)**:
    *   **测试集管理**: 支持添加和管理测试用例 (问题 + 标准答案)。
    *   **LLM-as-a-Judge**: 使用大模型作为裁判，自动评估 RAG 回答的质量。
    *   **评测指标**:
        *   **信实度 (Faithfulness)**: 回答是否忠实于检索到的上下文。
        *   **相关性 (Answer Relevance)**: 回答是否直接解决了用户的问题。
    *   提供雷达图和详细的评测报告。
*   **🔧 多模型支持 (Multi-Model Support)**:
    *   原生支持 OpenAI API。
    *   完美兼容阿里云 DashScope (通义千问 Qwen) API，解决了 Batch Size 限制和 Token 检查问题。

## 🛠️ 技术栈 (Tech Stack)

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Framework**: [LangChain](https://www.langchain.com/)
*   **Vector DB**: [ChromaDB](https://www.trychroma.com/)
*   **LLM API**: OpenAI API / Aliyun DashScope API
*   **Data Processing**: Pandas, PDFPlumber

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

确保已安装 Python 3.8+。

```bash
# 克隆仓库 (如果您还没有克隆)
git clone <your-repo-url>
cd Finance_RAG

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行应用

```bash
streamlit run src/web_ui/app.py
```

### 3. 系统配置

启动应用后，在左侧侧边栏进行配置：

1.  **API Key**: 输入您的 OpenAI API Key 或 阿里云 DashScope API Key。
    *   系统会自动检测环境变量 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`。
2.  **模型服务商 (Provider)**:
    *   选择 `OpenAI` 或 `Aliyun DashScope (Qwen)`。
3.  **模型名称**:
    *   根据需要修改 Chat Model (如 `gpt-3.5-turbo`, `qwen-plus`) 和 Embedding Model (如 `text-embedding-v1`)。
4.  点击 **"更新配置 & 重置系统"** 按钮使配置生效。

## 📂 项目结构 (Project Structure)

```
Finance_RAG/
├── data/                   # 数据存储 (临时文件, ChromaDB)
├── src/
│   ├── data_manager/       # 测试集管理模块
│   ├── eval_engine/        # 幻觉评测引擎 (LLM Judge)
│   ├── knowledge_base/     # 文档加载与向量库管理
│   ├── rag_engine/         # RAG 核心逻辑 (检索与生成)
│   └── web_ui/             # Streamlit 界面代码
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
└── ...
```

## 📝 常见问题 (FAQ)

*   **Q: 为什么上传文件时提示 "batch size is invalid"?**
    *   A: 通义千问 Embedding 接口限制单次批处理不超过 25 条。本项目已针对此进行了优化，将 `chunk_size` 限制为 10，请确保您运行的是最新版本代码。
*   **Q: 如何切换使用 Qwen-Max 或其他模型?**
    *   A: 在侧边栏的 "Chat Model Name" 输入框中直接输入模型名称 (如 `qwen-max`)，然后点击更新配置即可。

## 📄 License

[MIT License](LICENSE)
