# BookTalk — 技术规格
[English](SPEC.md) · **中文**

对上传的 PDF 书籍做语音问答。
流程：浏览器麦克风 → ASR → 在书上做 RAG → LLM 生成答案 → TTS → 浏览器播放。

本文档是实现的唯一依据。实现 agent 必须严格按照这里的文件树、API 契约和模块接口来做。
本规格没有写到的地方，选与设计理由一致的最简方案。

---

## 1. 目标与质量标准

- 能上传任意大小的 PDF 书籍（目标：处理 1000 页的书；必须能处理 ≥10 章）。
- 用语音提问；听到语音回答，同时看到文字和来源。
- 答案严格基于书的内容；书里没讲的就拒答。
- 检索必须在可测量的意义上优于朴素的固定长度分块（由评测脚本证明）。
- 延迟目标：问题转写完成后 ≤ 约 3.5 s 首次出声。
- 中英文都能用（书的语言和提问的语言都是）。
- 所有 OpenAI 调用都走一个可注入的客户端封装 → 完全可以离线测试。

## 2. 技术栈

- 后端：Python（≥3.9，开发机上的系统 python3 是 3.9.6 —— 所有依赖都必须支持它），FastAPI + uvicorn。不用 LangChain/LlamaIndex —— RAG 手写（体量小、可审查、可测试；这是一个要写进文档的有意的设计决定）。
- 前端：React 18 + TypeScript + Vite。不用 UI 框架（不用 MUI/AntD/Tailwind）—— 手写 CSS，配一套设计 token。不要组件库那种 "AI 模板" 的样子。
- 存储：SQLite（标准库 `sqlite3`），一个数据库文件 `data/booktalk.db`。embedding 以 float32 BLOB 存储。上传的 PDF 放在 `data/books/{book_id}.pdf`。
- OpenAI 模型（都可以用环境变量配置）：
  - ASR：`gpt-4o-mini-transcribe`（备选 `whisper-1`）
  - LLM：`gpt-4o-mini`（可用环境变量覆盖，例如 `gpt-4o`）
  - TTS：`gpt-4o-mini-tts`（备选 `tts-1`），音色 `alloy`，格式 mp3
  - Embedding：`text-embedding-3-small`（1536 维）
- 支持 `OPENAI_BASE_URL` 环境变量覆盖（方便走代理）。

## 3. 仓库结构

> **偏离说明 —— 项目结束时补写；下面的目录树保持原样。**
> 本节记录的是当初计划的样子，不是最终交付的样子，有三件事改变了它：§14 里由调研驱动
> 的修订，它们各自写明覆盖了什么；测试，从这里草拟的六个文件的样例长成了 326 个单元测试
> 和 43 个集成测试（见 [docs/TESTING.md](docs/TESTING.zh-CN.md)）；以及交互 UI 阶段，
> 它把前端整个换掉了 —— `Shell`、`Library`、`Toc`、`BookCard`、`SourceChip` 和 `Waveform`
> 都没了，`Shelf`、`ShelfBook`、`ShelfRail`、`Reader`、`BookPanel`、`SourceCard`、
> `Visualizer` 和一个 `hooks/` 目录取代了它们。后端加上了 §14 要求的模块（`ingest/toc.py`、
> `headings.py`、`markdown.py`、`vocab.py`、`enrich.py`、`prompts.py`；`index/rerank.py`；
> `rag/citations.py`、`prompts.py`；`speech/pcm.py`、`turns.py`、`phonetics.py`）。**实际建
> 成的目录树，逐个模块列出来，是 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.zh-CN.md) 开头
> 的「模块地图」** —— 当前的形态记录在那个文件里，不在这里。

```
booktalk/
  SPEC.md
  README.md                      # 文档阶段写的
  .gitignore                     # python, node, data/, .env
  docs/
    ARCHITECTURE.md
    TESTING.md
    AI_WORKFLOW.md
  backend/
    pyproject.toml               # PEP 621 元数据 + 依赖；用 `pip install -e .[dev]` 安装
    .env.example                 # OPENAI_API_KEY=, OPENAI_BASE_URL=, BOOKTALK_LLM_MODEL=, ...
    app/
      __init__.py
      main.py                    # FastAPI app 工厂、CORS、静态目录挂载、路由注册
      config.py                  # 从环境变量读的 Settings dataclass（不引 pydantic-settings）
      db.py                      # sqlite 连接辅助、schema DDL、启动时做迁移
      models.py                  # pydantic 请求/响应模型（只在 API 层用）
      services/
        openai_client.py         # AIClient protocol + OpenAIClient + FakeAIClient
      ingest/
        pdf.py                   # PyMuPDF 提取 → PageText[]（文本 + 字体 span + 页码）
        structure.py             # 章/节识别 → BookStructure 树
        chunker.py               # 结构感知的分块 → Chunk[]（子块 + 父块关联）
        ingestor.py              # 编排 解析→结构→分块→embedding→落库，带进度回调
      index/
        store.py                 # books/chunks/embeddings 的持久化；按书加载索引包
        dense.py                 # 在 numpy 矩阵上算 cosine top-k
        bm25.py                  # BM25 索引（rank-bm25）+ 分词器（CJK 用 jieba，拉丁文用正则）
        fusion.py                # reciprocal rank fusion
      rag/
        retriever.py             # 混合检索 → 子块 → 父块扩展 → 组装上下文
        rewrite.py               # 多轮对话的查询改写（没有历史时跳过）
        prompts.py               # 所有 prompt 模板，作为模块常量
        pipeline.py              # answer_stream(): 检索 → LLM 流式输出 → 断句 → TTS
      speech/
        sentences.py             # 面向流式文本的增量断句器（CJK + 拉丁标点）
      api/
        books.py                 # 上传/列表/状态/toc/删除
        asr.py                   # POST /api/asr
        chat.py                  # POST /api/chat (SSE)
      sessions.py                # 按 (session_id, book_id) 存的内存对话历史，LRU 有上限
    tests/
      conftest.py                # FakeAIClient fixture、临时 DB、app fixture、合成 PDF 构造器
      unit/
        test_structure.py
        test_chunker.py
        test_fusion.py
        test_sentences.py
        test_bm25_tokenizer.py
        test_dense.py
      integration/
        test_upload_and_ask.py   # 用 FakeAIClient 跑完整流程
        test_asr_endpoint.py
        test_error_paths.py      # 坏文件、未知的书、空问题、没有 key
    eval/
      make_fixture_book.py       # 从事实表生成测试用书 PDF（≥12 章）+ golden.json
      facts.py                   # 确定性的事实表（见 §10）
      run_eval.py                # 检索 + 答案评测，--config full|naive，--fake 用于离线冒烟
      README.md                  # 怎么跑，以及（之后补的）结果表
  frontend/
    package.json  vite.config.ts  tsconfig.json  index.html
    src/
      main.tsx  App.tsx
      lib/
        api.ts                   # §6 契约的带类型 fetch 封装
        sse.ts                   # 基于 POST 的 SSE 读取器（fetch + ReadableStream 解析）
        audioQueue.ts            # base64 mp3 分段的近似无缝顺序播放（WebAudio）
        recorder.ts              # MediaRecorder (audio/webm;codecs=opus) + analyser 电平
        types.ts
      components/
        Shell.tsx                # 两栏布局
        Library.tsx              # 上传拖放区 + 书卡片 + 入库进度
        Toc.tsx                  # 从 /toc 拿到的章节列表
        Conversation.tsx         # 消息列表、来源 chip、自动滚动
        MicButton.tsx            # 录音开关 + 实时电平环 + 各种状态
        Composer.tsx             # 文本输入兜底 + 麦克风
        SourceChip.tsx           # "Ch. 5 · p.123" → 可展开的引文
        Waveform.tsx             # 输入电平的实时可视化
      styles/
        tokens.css  base.css  components.css
    mock/
      server.mjs                 # node http 服务，用假数据实现 §6（前端不依赖后端也能开发）
  scripts/
    dev.sh                       # 同时起后端 (uvicorn) 和前端 (vite)
```

## 4. 入库设计（检索质量的核心）

### 4.1 PDF 解析 (`ingest/pdf.py`)
- PyMuPDF (`fitz`)。对每一页：`page.get_text("dict")` → 带字号/flags 的行。
- 输出 `PageText { page_no, lines: [{text, size, bold, y}] }`。内存要流式用：
  按页惰性解析；任何时刻都不要把整本书的完整 dict 一起放在内存里
  （只累积精简后的行记录）。

### 4.2 结构识别 (`ingest/structure.py`)
优先级顺序：
1. 内嵌 TOC：`doc.get_toc()`。如果有 ≥3 条 → 用第 1–2 层作为章/节
   （把 TOC 里的页码映射成边界）。
2. 启发式规则：一行是标题候选，条件是 (a) 匹配
   `^(chapter|part|book)\s+([0-9ivxlc]+|one|two|...)\b.*`（不分大小写），或 `^第[一二三四五六七八九十百0-9]+[章回部节].*`，
   或者 (b) 字号 ≥ 1.3 × 正文字号（正文字号 = 行字号的众数）并且 ≤ 80 字符并且在页面上部 25% 内。
   把候选按字号聚成章（更大）和节（更小）两级。
3. 兜底：一个标题都没有 → 造一个章 "Full text"。应用照常能用；UI 上不显示 TOC。

输出 `BookStructure { title, chapters: [{title, page_start, page_end, sections: [...] }] }`。
书名：PDF 元数据里的 title，没有就用文件名主干。

### 4.3 分块 (`ingest/chunker.py`)
- 在每个节内部（没有节就在章内部）：重建段落（合并行；空行或缩进作为断点；
  去掉页眉页脚 —— 在 >30% 的页面上重复出现的行）。
- 把段落打包成**子 chunk**：目标 450 token，硬上限 600，重叠 = 末尾 1 个段落。
  token 计数用 `tiktoken` 的 `cl100k_base`；tiktoken 不可用时退回 `len(text)//4`。
- **父块上下文**：节的文本，以子块为中心截到 1600 token。
- **上下文头**（和子块一起做 embedding、一起进 BM25 索引，不会重复喂给 LLM 两次）：
  `«{book_title}» — {chapter_title}[ — {section_title}]\n\n{chunk_text}`。
  理由：contextual retrieval 的廉价版本 —— 不用对每个 chunk 调 LLM 就能给它定位。
- chunk 元数据：`chapter_idx, chapter_title, section_title, page_start, page_end, char_span`。

### 4.4 Embedding + 持久化 (`ingest/ingestor.py`, `index/store.py`)
- 批量给子 chunk 生成 embedding（每次 API 调用 128 条输入），存成 float32 BLOB。
- 入库跑在后台线程；进度状态：`parsing → structuring → chunking → embedding (with pct) → ready | failed(reason)`。进度存在 DB 里，通过 `/status` 轮询。
- 幂等：重复上传同样的内容（文件的 sha256）直接返回已有的 ready 的书。

### 4.5 SQLite schema
```sql
books(id TEXT PK, title TEXT, filename TEXT, sha256 TEXT UNIQUE, pages INT,
      status TEXT, progress REAL, error TEXT, created_at TEXT);
chapters(book_id TEXT, idx INT, title TEXT, page_start INT, page_end INT,
         PRIMARY KEY(book_id, idx));
chunks(id TEXT PK, book_id TEXT, chapter_idx INT, section_title TEXT,
       page_start INT, page_end INT, text TEXT, header TEXT,
       parent_text TEXT, embedding BLOB, ord INT);
```
每本书的运行时索引包（LRU 缓存，最多 3 本）：numpy 矩阵（已归一化）、BM25 索引、
chunk 元数据列表。

## 5. 查询流程

### 5.1 ASR (`api/asr.py`)
- `POST /api/asr` multipart `audio`（MediaRecorder 产生的 webm/opus；Safari 的 mp4/m4a 也收）。
- 调用转写模型，返回 `{ "text": str, "duration_ms": int }`。
- 空结果或 whisper 幻觉出来的结果（strip 之后长度 < 2）用 422 `{detail}` 拒掉。

### 5.2 对话 (`api/chat.py` + `rag/pipeline.py`)
`POST /api/chat` JSON `{ book_id, session_id, question }` → SSE 流（`text/event-stream`，
用 StreamingResponse；不引 sse-starlette）。事件序列：

```
event: status   data: {"stage":"rewriting"|"retrieving"|"generating"|"speaking"}
event: question data: {"text": "<final question used for retrieval (post-rewrite)>"}
event: sources  data: {"sources":[{"chunk_id","chapter_idx","chapter_title","section_title","pages":[s,e],"preview":"<=200 chars"}]}
event: delta    data: {"text":"<token(s)>"}
event: audio    data: {"seq":0,"format":"mp3","b64":"..."}
event: done     data: {"answer":"...","timings":{"rewrite_ms","retrieve_ms","first_token_ms","first_audio_ms","total_ms"}}
event: error    data: {"message":"..."}
```

`answer_stream()` 内部的步骤：
1. **改写** (`rag/rewrite.py`)：只在 session 历史非空时做 —— 一次 `gpt-4o-mini` 调用，
   把问题加上最近 3 轮对话变成一个独立的查询。否则跳过（原样返回）。
2. **检索** (`rag/retriever.py`)：向量检索 top-24（在归一化矩阵上算 cosine）+ BM25 top-24
   → RRF (k=60) → 取前 8 个子块 → 按父块分组，保留 ≤5 个父块，上下文总量 ≤ 4500 token。
3. **生成**：流式 chat completion。系统 prompt（在 `prompts.py` 里）：只根据摘录回答；
   在行内以 (Chapter N) 的形式标注章节；如果摘录里没有答案，就说这本书没讲到 —— 绝不
   自己编；用提问的语言回答；口语语体，除非被要求展开否则 2–6 个短句；不要 markdown，
   不要列表（这段话是要念出来的）。
4. **按句流水线做 TTS** (`speech/sentences.py` + pipeline)：delta 流进来的同时，增量断句器
   吐出完整的句子（`.!?。!?…` 作边界，最少 25 字符 —— 比这短的和下一句合并；
   结束时把剩下的冲出去）。每个句子 → 一次 TTS 调用 → 一个 `audio` 事件。并发 2，靠 seq
   保序。第一句一完成就送去 TTS —— 延迟就是在这里省下来的。
5. 把这一轮写进 session 历史（内存，`sessions.py`，LRU 100 个 session × 20 轮次）。

### 5.3 错误处理
- `OPENAI_API_KEY` 缺失或无效 → ASR/chat 返回 503，附带能照着做的提示；`GET /api/health`
  报告 `{ok, openai_configured}`。
- 流中途 OpenAI 调用失败 → 发一个 `error` SSE 事件然后关闭；前端显示重试。

## 6. API 契约（前端 ⇄ 后端）

```
GET    /api/health                → {ok:true, openai_configured:bool}
POST   /api/books   (multipart "file")
                                  → 202 {book_id, status:"parsing"}  (or 200 existing ready book)
GET    /api/books                 → {books:[{book_id,title,pages,status,progress,created_at}]}
GET    /api/books/{id}/status     → {status, progress, error?}
GET    /api/books/{id}/toc        → {title, chapters:[{idx,title,page_start,page_end}]}
DELETE /api/books/{id}            → 204
POST   /api/asr     (multipart "audio")
                                  → {text, duration_ms}
POST   /api/chat    {book_id, session_id, question}
                                  → SSE per §5.2
```
上传约束：只收 `.pdf`，校验 `%PDF` 魔数，最大体积用环境变量配置
（默认 200 MB）。错误：FastAPI 默认的 `{detail}` 结构，配正确的状态码。
CORS：开发环境放行 `http://localhost:5173`。

## 7. 前端交互规格

> **偏离说明 —— 项目结束时补写；本节保持原样。**
> 下面这些交互 —— 上传和进度、麦克风的各种状态、带音频的流式输出、停止、空状态、
> 错误处理 —— 基本按描述做出来了。承载它们的界面没有，这里的响应式规则（侧栏收成一个
> 浮层面板）描述的是一个已经不存在的布局。暖纸色主题和它的 token 被换掉了两次 ——
> 先是 "Studio & Shelf" 方向，然后是
> **Studio Light**，它的配色、字体和玻璃效果规则在
> [frontend/DESIGN.md](frontend/DESIGN.zh-CN.md)。300px 侧栏的两栏布局没有了：应用现在
> 是两个页面 —— 一个放书的**书架**，和一个就是那本打开的书本身的**阅读页**，
> 一张封面在两者之间飞过去（书架见 [frontend/DESIGN-SHELF.md](frontend/DESIGN-SHELF.zh-CN.md)，
> 这次穿越和两个页面到底怎么工作见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.zh-CN.md) §8）。
> 来源是边注卡片，点了会把书翻到被引用的那段，而不是展开引文的 chip。音频是原始 PCM，
> 不是 mp3（§14.8），mock server 发的是生成的 PCM 音调，不是一段静音 mp3。有一处更正属于
> 这里而不属于 §14.6：**低置信度提示条不是由检索分数驱动的。** 它数的是答案里有多少句
> 能被事后的引用归属追溯到某一段原文，只有当一个实质性的答案（>80 字符）一句都没追溯到
> 时才出现 —— 这个分数在一本真实的书上两个方向都测错了（一个由四段被引用的原文正确回答
> 的问题得到 0.0017，一个书里从没提到过的主题得到 0.9953）。这个分数仍然在算，也仍然
> 随 `sources` 事件发出去。

布局：左侧栏 (300px) = 书库 + TOC；主栏 = 对话 + 输入区。
单一确定的视觉主题 —— 暖纸色的浅色主题：

```css
/* tokens.css */
--bg: #f6f1e7;          /* 暖纸色 */
--surface: #fffdf8;
--ink: #1f1b16;         /* 接近黑的暖色 */
--ink-soft: #6b6459;
--accent: #b4542a;      /* 赤陶色 */
--accent-ink: #fff;
--line: #e3d9c8;
--radius: 10px;
font-display: "Fraunces", Georgia, serif;   /* 书名、标题（Google Fonts）*/
font-body: "Inter", -apple-system, sans-serif;
```

> **设计权威说明（采用 skill 之后补写）：** `.claude/skills/` 下的设计 skill
> （`impeccable-craft`、`minimalist-ui`、`redesign-existing-projects`）
> 是**所有**前端工作（构建和打磨）中视觉/交互质量的约束性权威，与上面的 token 细节冲突时
> 以它们为准 —— 特别是：正文字体不能用 Inter
> （按 `minimalist-ui` 选一个有辨识度的无衬线体）、阴影几乎不可见、只用一个强调色、
> 所有组件状态（hover/focus/active/disabled/loading/error/empty）都要做。

关键交互：
- 上传：拖放或者点击；入库过程中书卡片显示阶段 + 进度条
  （每 1s 轮询 /status）；ready 的书显示书名、页数、章数。
- 麦克风：点击开始（按钮变成一个脉动的环，由 analyser 拿到的真实输入电平驱动），
  再点停止 → "Transcribing…" → 转写文本作为用户消息出现
  → SSE 答案流式进来，带一个不显眼的文字光标；音频分段边到边播。
- 说话状态：助手消息上有一个小的均衡器动画；一个停止按钮取消播放并中断流
  （AbortController）。
- 来源：答案下面的 chip（"Ch. 3 · pp. 41–44"）；点击展开摘录预览。
- 输入区还有一个文本输入框（无障碍 + 没有麦克风也能测）。回车发送。
- 空状态：没有书 → 居中的上传主区，配一句话说明这条流程；
  书已经 ready 但还没有消息 → 3 个建议问题（通用的，例如 "What is Chapter 1 about?"）。
- 错误：toast + 在失败的那条消息上就地重试；麦克风权限被拒时说明浏览器里怎么改。
- 没有深色模式（单一确定的外观）。响应式向下支持到 900px（侧栏收成一个浮层面板）。

音频播放 (`audioQueue.ts`)：WebAudio —— 用 `decodeAudioData` 解码每个 base64 mp3 分段，
在同一条 `AudioContext` 时间轴上把 buffer 首尾相接地排好
（next.start(max(ctx.currentTime, lastEnd))）；暴露 stop() 和一个给均衡器用的 `onLevel` 回调。
处理 iOS/Safari 的自动播放限制：在点击麦克风这个手势里创建 AudioContext。

Mock server (`frontend/mock/server.mjs`)：用假数据实现 §6，音频事件用一小段静音的
base64 mp3；`npm run mock` 在 :8000 上跑它，这样不需要后端、也不需要 API key 就能完整
操作 UI。

## 8. 配置

`backend/.env`（在 `config.py` 里通过 os.environ + 可选的 python-dotenv 加载）：
```
OPENAI_API_KEY=
OPENAI_BASE_URL=            # 可选
BOOKTALK_LLM_MODEL=gpt-4o-mini
BOOKTALK_ASR_MODEL=gpt-4o-mini-transcribe
BOOKTALK_TTS_MODEL=gpt-4o-mini-tts
BOOKTALK_TTS_VOICE=alloy
BOOKTALK_EMBED_MODEL=text-embedding-3-small
BOOKTALK_MAX_UPLOAD_MB=200
BOOKTALK_DATA_DIR=./data
```

## 9. AIClient 封装 (`services/openai_client.py`)

```python
class AIClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def transcribe(self, audio: bytes, filename: str) -> str: ...
    def chat_stream(self, messages: list[dict], **kw) -> Iterator[str]: ...  # 产出文本 delta
    def chat(self, messages: list[dict], **kw) -> str: ...
    def tts(self, text: str) -> bytes: ...  # mp3
```
`OpenAIClient` 用 `openai` SDK 实现（重试 2 次，超时 60s）。
`FakeAIClient`（测试和 `--fake` 评测用）：基于哈希的确定性 embedding
（用 sha1(text) 播种的 RandomState，做单位归一化 —— 相似的文本不会得到相似的向量，
测试不能假设语义相似性，只能假设同样的文本结果确定），
固定的 chat 流（回显一个模板，里面写上检索到的章），1 秒的静音 mp3，
transcribe 返回一个固定字符串。通过 FastAPI 依赖注入（`app.state.ai`）。

## 10. 测试用书 + 标准答案集 (`backend/eval/`)

- `facts.py`：一本虚构的非虚构书的确定性事实表，书名
  “A Field Guide to the Meridian Archipelago” —— 12 章（每章一个岛），每章约 8 条结构化
  事实（创建者姓名、年份、人口、地标、当地菜、距离……）。事实全是编的
  （无污染：LLM 不可能从预训练里知道它们 —— 这正是关键；要写进文档）。
- `make_fixture_book.py`：用 PyMuPDF 的 insert 接口或者 `reportlab` 渲染这本 PDF
  （选 reportlab；加进 dev 依赖）：标题页，每章标题 “Chapter N — <Island>”
  用 20pt 粗体，正文 11pt，每章几个段落把事实织进行文里，
  每章 ≥2 页。同时产出 `golden.json`：
  `[{q, expected_answer_substrings:[...], chapter_idx}]` —— 每章约 4 个问题（共 48 个），
  来自同一张事实表，包括一些跨章的问题和一些
  “书里没有” 的问题（期望拒答）。
- `run_eval.py`：
  - 用 `full`（本规格）或者 `naive`
    （800 字符固定 chunk、没有上下文头、只做向量检索）配置把测试用书入库 —— 后者是消融实验的基线。
  - 检索指标：recall@5 / recall@8（检索到的 chunk 的 chapter_idx 对上就算命中）、
    MRR。答案指标：expected_answer_substrings 的子串命中率；
    书外问题的拒答正确率。
  - `--fake` 用 FakeAIClient 跑整套评测脚本（验证管路通不通，不验证质量）。
  - 往 stdout 和 `eval/report.md` 输出一个 markdown 表格。

## 11. 测试策略

- 单元测试（离线，不联网）：在测试里合成的 PDF 上做结构识别（用
  PyMuPDF 构造：带指定字号的页面）、分块的不变量（token 上限被遵守、没有空 chunk、
  元数据正确、页眉页脚被剥掉）、RRF 排序的算法、增量断句器
  （拉丁文 + CJK + 结尾冲刷）、分词器（CJK 的 jieba 路径）。
- 集成测试（离线）：TestClient + FakeAIClient：上传一个小的合成 PDF → 轮询到 ready →
  toc 正确 → chat SSE 产出完全一致的事件序列（status→sources→delta+→audio+→done）→
  错误路径（魔数不对 400、超大 413、未知的书 404、没有 key 503）。
- 前端：`npm run build` 必须零 TS 错误通过；用 vitest 给 `sse.ts` 的解析器和
  `audioQueue` 的保序逻辑写单元测试（mock 掉 AudioContext）。
- 答案质量评测：§10 的脚本用真 key 跑（之后做；数字写进 README）。
- 手工端到端检查清单放在 docs/TESTING.md（麦克风权限、Safari、大书、停止按钮）。

## 12. 不做的事

- 不做认证/多用户，不做部署设施（只在本地跑；README 里说明 `scripts/dev.sh`）。
- 不做深色模式。UI 文案不做多语言（UI 用英文；答案跟随提问的语言）。
- 默认不用 cross-encoder 重排模型（延迟）；LLM 重排放在
  `BOOKTALK_RERANK=1` 环境变量后面：一次 gpt-4o-mini 调用给 top-8 打分 → 保留 top-5 个父块。

## 13. 依赖（版本要和 Python 3.9 兼容）

后端运行时：fastapi, uvicorn[standard], python-multipart, openai (≥1.40),
pymupdf, numpy（3.9 的 wheel 需要的话用 <2）, rank-bm25, jieba, tiktoken, python-dotenv。
开发：pytest, httpx, reportlab。
前端：react, react-dom, typescript, vite, @vitejs/plugin-react, vitest (dev)。

## 14. v2 修订（由调研驱动；在写明的地方覆盖前面的章节）

来源：五个 agent 的 GitHub 调研（见 docs/RESEARCH.md）。作为 v1 后端之上的一个升级
阶段执行。每一条都写明它覆盖哪一节。

### 14.1 提取（覆盖 §4.1）
- 用 `pymupdf4llm`（已验证能装在 py3.9 上，约 50 MB）作为文本提取层，
  通过它的 `hdr_info=` 钩子注入我们自己的结构识别。在一本真实的
  244 页的书上测到的收益：原始 `get_text()` 有 129 个连字字符和 68 处行尾断词连字符，
  pymupdf4llm 是 0 和 0 —— 去掉断词连字符直接就是 BM25 的收益。README 里要注明 AGPL-3.0。
- 扫描版 PDF 预检：统计前 10 页的每页字符数；纯图片的 PDF
  大约是 0（原生数字版约 1400+）。用明确的 UI 提示拒掉（OCR 不在范围内）。
- 保留重复行的页眉页脚剥离（跨页检测，不是裁掉页边）。

### 14.2 结构识别的阶梯（覆盖 §4.2）
第 1 层：内嵌 TOC (`get_toc()`)。第 2 层：印刷目录页解析器（在书自己的目录页上跑一个
状态机；用页码单调递增来校验条目）。第 2.5 层：
标题家族投票，从 RAGFlow 的 `bullets_category` 移植：给若干正则家族
（`Chapter <roman|word|number>`、`PART ONE..`、`^\d{1,2}[.、 ]`、`第..[章节部]`、markdown `#`）
在所有行上打分，取得分最高的那个家族。第 3 层：字号直方图（v1 的启发式）。
另外：当 TOC 存在但没有可用的层级时，在入库时调一次 LLM 来分配层级
（RAGFlow 的 `assign_toc_levels`：严格 JSON `[{level,title}]`，保持顺序）。
绝不让一个孤立的标题自己成为一个 chunk。

### 14.3 分块（覆盖 §4.3）
移植 Docling `HybridChunker` 的三遍算法（约 150 行，MIT），用 `semchunk`
（已验证支持 py3.9）作为叶子层的切分器：(a) 维护 `heading_by_level`，遇到一个标题就把
层级 ≥ 它的键弹出去，这样每个 chunk 都带着完整的祖先路径；(b) 按**加上上下文之后**的
token 数做预算 —— `available = max_tokens - header_tokens`；(c) 只有在标题路径完全一致时
才合并相邻的过小的 chunk。子块目标 256–320 token，
**零重叠**（Chroma 的证据：重叠有害；200/0 的递归切分拿到 88.1% recall）。
护栏：header ≤ chunk token 数的 25%，正文 ≥ 下限（宁可丢掉 header 也不要饿着正文）；
一个节的第一个 chunk 省掉 header；过长的标题路径从头部截断（保留最具体的那几层）。

### 14.4 上下文增强升级（扩展 §4.3）
配置 `BOOKTALK_CONTEXTUAL=1`（默认开）：用 LLM 为每个 chunk 生成定位性的上下文
（Anthropic 的 contextual retrieval；公开的数字是检索失败率 5.7%→2.9%），生成上下文的
prompt 只覆盖这个 chunk 所在的**章**（不是整本书），为的是控成本。入库时并发
调用；进度反映在入库状态里。这个开关关掉时，静态的路径 header 就是兜底 ——
它也是一个消融实验分组。

### 14.5 章节摘要节点（扩展 §4.4；RAPTOR 的简化版）
每章一条 LLM 摘要（再加一条全书级的），作为可检索的节点索引，打上
`level` 标签。这能回答扁平 chunk 被证明会漏掉的主题性问题（"第 7 章讲的是什么"）。
完整的 RAPTOR 聚类被否掉（要 UMAP/numba，只有 2–3 个点的提升，而且和一棵真实的 TOC 树重复）。
摘要同时在 UI 里当作 TOC 的内容简介。

### 14.6 检索（覆盖 §5.2 第 2 步）
向量检索 top-40 + BM25 top-40 → 加权 RRF（在评测里扫 k∈{10,20,60} × 各种权重分配；
TREC 的 k=60 假设的是百万级文档的语料）→ **FlashRank 重排**（`ms-marco-TinyBERT-L-2-v2`，
`flashrank` + `onnxruntime==1.19.2` —— 已在这台机器上验证：5 MB 模型，20 段 7.6 ms，
cp39 wheel 能解析出来；启动时预热）作用在 top-20 的**子** chunk 上（max_length
512 会把父块截断）→ 取前 8 → 只对分数超过阈值的 chunk 做父块扩展 →
按章分组，按阅读顺序排序，合并连续的片段，去重 →
上下文 ≤4500 token。除了排名之外还保留每一路信号归一化后的分数：它们驱动
拒答闸门和一个低置信度 UI 提示条（"Context relevance is low — double-check
this answer"，kotaemon 的做法）。

### 14.7 引用改为事后计算（覆盖 §5.2 第 3 步里行内的 "(Chapter N)"）
生成用的 prompt 不再强制行内的引用标记（念出来的文本保持干净）。答案生成完之后：
切成句子，批量做 embedding，用混合相似度（tkweight 0.1 / vtweight 0.9）把每个句子和
检索到的 chunk embedding 做匹配，阈值从 0.63 递降 → ×0.8 → 下限 0.3，每句 ≤4 个来源
（RAGFlow 的 `insert_citations`）。这一步和 TTS 的尾部并发跑；新增 SSE 事件 `citations`
（`{turn_id, map:[{sentence_idx, chunk_ids}]}`），在 `done` 之前发出。对首次出声的时间
零增加。

### 14.8 语音流程（覆盖 §5.2 第 4 步和 audio 事件的格式）
- TTS 输出用 `response_format="pcm"`（24 kHz，s16le，单声道）。audio SSE 事件：
  `{"turn_id","seq","format":"pcm","sample_rate":24000,"b64"}`。不用 mp3，没有解码这一步。
- 断句分两个阶段（延迟最优）：第一个片段在任何软分隔符
  （`,;:.!?…`）处、已有 ≥10 字符时就发出，超过约 30 个词后在下一个空格处强制发出
  （stream2sentence 的参数）；之后的每一段都要求真正的句子边界并且 ≥20
  字符（LiveKit 的 `min_sentence_len`）。把 LiveKit 的 `split_sentences` 复制进仓库（约 60 行，
  Apache-2.0，在文件里写明出处）+ pipecat 的非空白前瞻（"$29." 对 "$29. Next"）
  + 12 字符的尾部保护 + 绝不在数字中间切。用表驱动的单元测试。
- 背压：句子→TTS 的队列上限 1；未播放的音频超过 5 s 时暂停合成。
  取消一轮只浪费一句话的花费，不是整个答案。
- 每个 SSE 事件都带 `turn_id`。取消：客户端中断流并向
  `/api/chat/turns/{turn_id}/cancel` POST `{played_seq}`；服务端取消 LLM/TTS 任务，
  清空队列，把存下来的历史截断到实际播放出去的那些句子（vocode 的做法），
  这样后续提问看到的上下文是准确的。

### 14.9 ASR 加固（扩展 §5.1；语音场景的差异化点）
由 ASR 引起的 RAG 失败里，87–96% 是命名实体被识别坏了（arXiv 2608.22872），而一本书
免费给了我们它的实体词表：入库时抽出高频的专有名词/首字母大写的
token；把排名靠前的实体传进 ASR 请求的 prompt 参数
（要核实 OpenAI 当前的语义）；ASR 之后，在书的词表上跑一层 Double Metaphone 模糊匹配
（纯 Python），在检索之前修掉被转写错的实体。
测试用书里编造的名字让这一点可以直接评测（ASR 污染的消融实验分组）。

### 14.10 评测升级（覆盖 §10/§11 的指标和分组）
- 指标：recall@{1,3,5,10}、MRR、token 级的 IoU（Chroma 风格的精确率信号）、
  答案子串命中、拒答正确率；按类别的 CSV + 排名分布。
- 类别：单一事实 / 跨章综合 / 章节摘要 / 书里没有
  （拒答）/ ASR 污染的音近变体。
- 消融实验分组：A 朴素的固定 800 且只做向量检索 · B +结构感知的分块+标题路径 ·
  C +混合 RRF · D +重排 · E +章节摘要 · F +LLM 上下文增强。
  再加上 (k, weight) 的 RRF 扫描。
- ragas/deepeval **被否掉**并给出理由（两者在 py3.9 上都能解出依赖，但分别会拉进 langchain/datasets/
  scipy 和 grpcio/otel/posthog；LLM 评判的分数对于衡量小的消融差值是很差的工具）。
  改成手写的忠实度评判（约 30 行 prompt，跑在我们自己的客户端上）。

### 14.11 明确否掉的方案（写进 README）
Late chunking（需要 token 级的 embedding；OpenAI 只返回池化后的向量）。HyDE
（在首次出声这条关键路径上多一次完整的 LLM 往返 + 实体幻觉的风险）。
完整的 RAPTOR（见 14.5）。把语义分块作为主路径（NAACL 2025 Qu/Tu/Bao + Chroma：
在真实文档上，相对固定长度分块，这个成本不划算）—— 只作为可选的消融实验分组。

### 14.12 新增依赖（扩展 §13）
后端：pymupdf4llm, semchunk, flashrank, onnxruntime==1.19.2, 一个 Double Metaphone
实现（小的纯 python 依赖，或者把代码复制进仓库）。前端（已经交代给前端构建
agent）：radix-ui@1.6.7, eventsource-parser@4.1.0, use-stick-to-bottom@1.1.6,
@fontsource-variable/literata；PCM 流播放器；Tufte 式边注的引用布局；
复古纸色配色（#f5f1e6 / #fffcf5 / #4a3f35 / #7d6b56 / #dbd0ba / 强调色 #a67c52，
高亮 #FBDA83）；圆角 0.25rem；偏移阴影。

## 15. 每个阶段的完成标准

- 后端 agent：§6 的所有端点都实现；`pytest tests/unit` 在本地全绿；
  没有 key 也能启动（health 报 openai_configured:false）—— 单元测试和代码一起写。
- 前端 agent：`npm run build` 通过；UI 能完整地跑在 `mock/server.mjs` 上。
- 测试 agent：集成测试套件离线全绿；覆盖 §6 的每一个端点 + SSE 序列。
- 评测 agent：测试用书能生成（≥12 章，≥25 页）；`run_eval.py --fake` 能跑完。
- 文档 agent：README（快速上手 ≤10 条命令、mermaid 架构图、设计
  决策含被否掉的方案、评测表格占位）、ARCHITECTURE.md、
  TESTING.md、AI_WORKFLOW.md 骨架。
