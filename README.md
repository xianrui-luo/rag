# Folder RAG Local / 本地文件夹 RAG

Local folder-based RAG app with Gradio, Chroma, SQLite, local embeddings, and cloud generation.

基于本地文件夹的 RAG 应用，使用 Gradio、Chroma、SQLite、本地 embedding 模型，以及云端生成接口。

## Features / 功能特性

- Local folder indexing for `PDF`, `DOCX`, `MD`, `TXT`
- Support incremental refresh for added, updated, and deleted files
- Support full rebuild for a knowledge base
- Use local embedding model via `sentence-transformers`
- Add section-aware chunking for academic papers
- Suppress reference sections during indexing by default
- Combine vector retrieval, FTS/BM25 lexical retrieval, neighbor expansion, query rewriting, and reranking
- Use OpenAI-compatible cloud API for answer generation
- Provide Gradio UI for indexing and Q&A

- 支持本地文件夹索引：`PDF`、`DOCX`、`MD`、`TXT`
- 支持增量刷新，自动处理新增、修改、删除文件
- 支持知识库完整重建
- 使用 `sentence-transformers` 本地 embedding 模型
- 针对学术论文增加章节感知分块
- 默认抑制参考文献分块进入索引
- 结合向量检索、FTS/BM25 词法检索、邻近块扩展、问题改写与重排
- 使用 OpenAI 兼容云接口完成答案生成
- 提供 Gradio 图形界面用于建库和问答

## Quick Start / 快速开始

### Linux / macOS

1. Create a virtual environment and install dependencies.
2. Copy env config.
3. Set `OPENAI_API_KEY` and other values if needed.
4. Start the app.
5. Open the local Gradio URL shown in the terminal.

1. 创建虚拟环境并安装依赖。
2. 复制环境变量模板。
3. 填写 `OPENAI_API_KEY` 和其他必要配置。
4. 启动应用。
5. 打开终端输出的本地 Gradio 地址。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 app.py
```

## Windows + Conda

This project is currently bound to `D:\anaconda\envs\normal`.

当前项目已绑定到 `D:\anaconda\envs\normal` 环境。

### First-time Setup / 首次准备

1. Copy `.env.example` to `.env`
2. Fill at least these values:
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`
   - `OPENAI_MODEL`
3. Install dependencies if needed

1. 将 `.env.example` 复制为 `.env`
2. 至少填写以下配置：
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`
   - `OPENAI_MODEL`
3. 如未安装依赖，请先安装

```bat
copy .env.example .env
D:\anaconda\envs\normal\python.exe -m pip install -r requirements.txt pyinstaller
```

### Start / 启动

Double-click `run_windows.bat` in the project root, or run:

直接双击项目根目录下的 `run_windows.bat`，或执行：

```bat
run_windows.bat
```

The launcher will:

- switch to the project root
- verify `D:\anaconda\envs\normal\python.exe`
- verify `.env`
- run `app.py` with the fixed Conda Python

启动器会：

- 自动切换到项目根目录
- 检查 `D:\anaconda\envs\normal\python.exe`
- 检查 `.env`
- 使用固定的 Conda Python 运行 `app.py`

## Workflow / 使用流程

- Enter a knowledge base name and folder path
- Click `Refresh Index` for incremental updates
- Click `Rebuild Index` for a full reset and re-index
- Ask questions and review cited source chunks

- 输入知识库名称和文件夹路径
- 点击 `Refresh Index` 执行增量刷新
- 点击 `Rebuild Index` 执行完整重建
- 输入问题并查看引用到的分块来源

## Academic Paper Mode / 学术论文模式

- PDF papers are cleaned before chunking to reduce broken line joins and running headers
- The chunker prefers section and paragraph boundaries, then falls back to sliding windows for long blocks
- Reference sections are excluded by default to reduce noisy citations in retrieval
- Retrieved sources now include section name and page range metadata
- Follow-up questions can be rewritten into standalone retrieval queries before search
- Top hits can pull same-section neighboring chunks into the final answer context
- Lexical retrieval now uses SQLite FTS5 with BM25 scoring instead of full Python scans

- PDF 论文在分块前会先做基础清洗，减少断行拼接和页眉干扰
- 分块优先遵循章节和段落边界，过长文本再退回滑窗切分
- 默认排除参考文献章节，减少检索时的引用噪声
- 检索结果会展示章节名和页码范围
- 连续追问会先被改写成独立检索问题，再进入检索流程
- 命中的高相关块会补充同章节邻近块进入最终回答上下文
- 词法检索已改为 SQLite FTS5 + BM25，不再做 Python 全量扫描

## Retrieval Settings / 检索设置

- `RERANKER_MODEL`: local reranker model name
- `RETRIEVAL_CANDIDATES`: number of vector candidates before fusion
- `LEXICAL_CANDIDATES`: number of lexical candidates before fusion
- `HISTORY_TURNS`: number of recent dialogue turns used for query rewriting and answer generation
- `MAX_CONTEXT_BLOCKS`: maximum number of final context blocks after neighbor expansion
- `NEIGHBOR_EXPANSION_WINDOW`: how many same-section neighboring chunks to add around a hit
- `ENABLE_HYBRID_RETRIEVAL`: enable vector + lexical fusion
- `ENABLE_RERANKER`: enable local reranking
- `ENABLE_QUERY_REWRITE`: rewrite follow-up questions into standalone retrieval queries
- `EXCLUDE_REFERENCES`: skip references during indexing

- `RERANKER_MODEL`：本地重排模型名称
- `RETRIEVAL_CANDIDATES`：融合前的向量召回候选数
- `LEXICAL_CANDIDATES`：融合前的词法召回候选数
- `HISTORY_TURNS`：用于问题改写和回答生成的最近对话轮数
- `MAX_CONTEXT_BLOCKS`：邻近扩展后最终送入回答阶段的上下文块上限
- `NEIGHBOR_EXPANSION_WINDOW`：命中块两侧同章节邻近块的扩展窗口大小
- `ENABLE_HYBRID_RETRIEVAL`：是否启用向量 + 词法融合
- `ENABLE_RERANKER`：是否启用本地重排
- `ENABLE_QUERY_REWRITE`：是否将追问改写成独立检索问题
- `EXCLUDE_REFERENCES`：索引时是否排除参考文献

## Notes / 注意事项

- Changing `EMBEDDING_MODEL`, `CHUNK_SIZE`, or `CHUNK_OVERLAP` requires `Rebuild Index`
- The first run downloads the embedding model locally
- Answer generation still calls your configured cloud API
- `run_windows.bat` is a stable Windows launcher, not a standalone `.exe`

- 修改 `EMBEDDING_MODEL`、`CHUNK_SIZE`、`CHUNK_OVERLAP` 后，需要执行 `Rebuild Index`
- 首次运行会在本地下载 embedding 模型
- 生成答案时仍会调用你配置的云端 API
- `run_windows.bat` 是更稳的 Windows 启动器，不是独立打包后的 `.exe`

## Optional Next Step / 可选后续

If you later want a real Windows executable, you can package from the current environment with PyInstaller.

如果你之后想要真正的 Windows 可执行文件，可以继续基于当前环境使用 PyInstaller 打包。
