# 知流 RAG（RagProject）

> 一个基于 **LangGraph + FastAPI** 的企业知识库 RAG 系统：把 PDF/Markdown 文档解析、切分、向量化并入库，再以多路召回 + 融合 + 重排 + 大模型生成的方式回答用户问题，支持 SSE 流式输出与多轮历史。

系统拆成两条独立的 LangGraph 流水线：

- **导入流水线（import_process）**：上传文件 → 解析为 Markdown → 图片处理 → 文档切分 → 主体名识别 → BGE-M3 向量化 → 写入 Milvus。
- **查询流水线（query_process）**：主体确认 / query 改写 → 多路召回（向量 / HyDE / 联网搜索）→ RRF 融合 → 重排 → 组装 Prompt → LLM 生成答案。

两条流水线各自由一个 FastAPI 服务对外提供 HTTP/SSE 接口，并附带可直接使用的 HTML 页面（导入页、对话页、监控页）。

---

## 技术栈

| 类别 | 使用 |
|---|---|
| Web 服务 | FastAPI + Uvicorn，SSE 流式输出 |
| 编排 | LangGraph（`StateGraph`）、LangChain |
| 向量库 | Milvus（`pymilvus[model]`） |
| 向量 / 重排模型 | BGE-M3（`FlagEmbedding`）、BGE Reranker（`transformers` / `torch`） |
| 文档解析 | MinerU（PDF → Markdown） |
| 对象存储 | MinIO（存放上传原文件与图片） |
| 会话历史 | MongoDB |
| 知识图谱（可选） | Neo4j |
| 大模型 | OpenAI 兼容接口（通义千问 / DashScope 等），Visual 模型做图片摘要 |
| 联网搜索 | 百炼 / DashScope MCP |
| 其它 | loguru、modelscope（模型下载）、python-dotenv |

要求 **Python ≥ 3.14**，使用 [uv](https://github.com/astral-sh/uv) 管理依赖。

---

## 目录结构

```
app/
├── import_process/                 # 文档导入流水线
│   ├── agent/
│   │   ├── main_graph.py           # 导入状态图（kb_import_app）
│   │   ├── state.py                # ImportGraphState
│   │   └── nodes/                  # entry / pdf_to_md / md_img / document_split
│   │                               #   / item_name_recognition / bge_embedding / import_milvus
│   ├── api/import_server.py        # FastAPI 导入服务（:8000）
│   └── page/import.html            # 导入前端页面
├── query_process/                  # 查询问答流水线
│   ├── agent/
│   │   ├── main_graph.py           # 查询状态图（query_app）
│   │   ├── state.py                # QueryGraphState
│   │   └── nodes/                  # item_name_confirm / search_embedding / search_embedding_hyde
│   │                               #   / web_search_mcp / rrf / rerank / answer_output
│   ├── api/query_server.py         # FastAPI 查询服务（:8001）
│   ├── sse/                        # SSE 接入示例
│   └── page/                       # chat.html / query_monitor.html
├── clients/                        # Milvus / MinIO / MongoDB / Neo4j 客户端封装
├── conf/                           # 各服务配置（从 .env 读取）
├── lm/                             # embedding / LLM / reranker 调用封装
├── core/                           # 日志、Prompt 加载
├── tool/                           # 模型下载脚本
└── utils/                          # SSE、任务状态、路径、限流、向量归一化等
prompts/                            # 各阶段提示词模板
test/                               # 图导入/查询的手工测试脚本
```

---

## 数据流

**导入：**

```
node_entry ──(pdf)──▶ node_pdf_to_md ──▶ node_md_img ──▶ node_document_split
           └─(md)──────────────────────▶ node_md_img ──▶ ...
node_document_split ──▶ node_item_name_recognition ──▶ node_bge_embedding ──▶ node_import_milvus ──▶ END
```

**查询：**

```
node_item_name_confirm ──(需要澄清)──▶ node_answer_output
        └─▶ node_search_embedding ─┐
        └─▶ node_search_embedding_hyde ─┤──▶ node_rrf ──▶ node_rerank ──▶ node_answer_output ──▶ END
        └─▶ node_web_search_mcp ────┘
```

---

## 依赖的外部服务

运行前需要准备好（地址/账号都在 `.env` 中配置）：

- **Milvus**：向量数据库（切片集合、实体名集合等）
- **MongoDB**：会话历史
- **MinIO**：原文件与图片对象存储
- **Neo4j**（可选）：知识图谱
- **OpenAI 兼容的大模型服务**：生成答案、图片摘要、主体识别
- **MinerU 服务**：PDF 解析
- **DashScope / 百炼**：联网搜索 MCP

---

## 配置

项目通过根目录 `.env` 读取配置（`.env` 已在 `.gitignore` 中，不会提交）。请自行创建 `.env` 并填写，主要变量分组如下：

```env
# —— 大模型（OpenAI 兼容）——
OPENAI_BASE_URL=
OPENAI_API_KEY=
LLM_DEFAULT_MODEL=
LLM_DEFAULT_TEMPERATURE=
VL_MODEL=

# —— BGE-M3 向量模型 ——
BGE_M3_PATH=            # 本地模型路径
BGE_M3=                 # 模型仓库标识
BGE_DEVICE=cpu          # 或 cuda:0
BGE_FP16=0

# —— BGE Reranker 重排模型 ——
BGE_RERANKER_LARGE=
BGE_RERANKER_DEVICE=cpu
BGE_RERANKER_FP16=0

# —— Milvus ——
MILVUS_URL=
CHUNKS_COLLECTION=
ENTITY_NAME_COLLECTION=
ITEM_NAME_COLLECTION=
EMBEDDING_DIM=
MILVUS_METRIC_TYPE=IP
MILVUS_MIN_COSINE_SCORE=0.0

# —— MongoDB ——
MONGO_URL=
MONGO_DB_NAME=

# —— MinIO ——
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET_NAME=
MINIO_IMG_DIR=
MINIO_SECURE=0

# —— MinerU（PDF→MD）——
MINERU_BASE_URL=
MINERU_API_TOKEN=
MINERU_MODEL_SOURCE=

# —— 联网搜索 MCP（DashScope / 百炼）——
MCP_DASHSCOPE_BASE_URL=

# —— Neo4j（可选）——
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=

# —— 日志 ——
LOG_CONSOLE_ENABLE=1
LOG_CONSOLE_LEVEL=INFO
LOG_FILE_ENABLE=1
LOG_FILE_LEVEL=INFO
```

---

## 安装

```bash
# 安装依赖（已配置国内源 + CPU 版 torch 源）
uv sync
```

### 下载模型

```bash
# BGE-M3 向量模型
python app/tool/download_bgem3.py

# BGE Reranker 重排模型
python app/tool/download_reranker.py
```

> 两个脚本中的下载目录是示例路径，请按需修改为你的 `BGE_M3_PATH` / `BGE_RERANKER_LARGE` 对应位置。

---

## 运行

```bash
# 导入服务： http://127.0.0.1:8000
#   导入页面 http://127.0.0.1:8000/import
#   Swagger  http://127.0.0.1:8000/docs
python -m app.import_process.api.import_server

# 查询服务： http://127.0.0.1:8001
#   对话页面 http://127.0.0.1:8001/chat.html
#   健康检查 http://127.0.0.1:8001/health
python -m app.query_process.api.query_server
```

### 主要接口（查询服务）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/chat.html` | 对话页面 |
| POST | `/query` | 发起提问（可 `is_stream` 走 SSE） |
| GET | `/sse/...` | SSE 流式结果 |
| GET/POST | 历史对话相关 | 查看 / 清空会话历史 |

---

## 说明

本项目面向学习与内部实践，接口与节点实现贴近业务场景（含商品/实体名识别等定制化步骤）。二次使用请按自身数据与模型服务调整 `app/conf/*` 与 `prompts/`。
