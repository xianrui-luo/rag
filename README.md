# Folder RAG Local / 本地文件夹 RAG

Local folder-based RAG app with Gradio, Chroma, SQLite, local embeddings, and cloud generation.

基于本地文件夹的 RAG 应用，使用 Gradio、Chroma、SQLite、本地 embedding 模型，以及云端生成接口。

## Features / 功能特性

- Local folder indexing for `PDF`, `DOCX`, `MD`, `TXT`
- Support incremental refresh for added, updated, and deleted files
- Support full rebuild for a knowledge base
- Use local embedding model via `sentence-transformers`
- Use OpenAI-compatible cloud API for answer generation
- Provide Gradio UI for indexing and Q&A

- 支持本地文件夹索引：`PDF`、`DOCX`、`MD`、`TXT`
- 支持增量刷新，自动处理新增、修改、删除文件
- 支持知识库完整重建
- 使用 `sentence-transformers` 本地 embedding 模型
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

If you later want a real Windows executable, you can package from the current `normal` environment with PyInstaller.

如果你之后想要真正的 Windows 可执行文件，可以继续基于当前 `normal` 环境使用 PyInstaller 打包。
