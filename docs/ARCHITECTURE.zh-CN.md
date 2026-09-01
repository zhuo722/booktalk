# 架构

[English](ARCHITECTURE.md) · **中文**

BookTalk 如何把一份 PDF 变成一本可以对话的书，以及每个部件为什么长成现在这个样子。约束性契约是 [SPEC.md](../SPEC.zh-CN.md) —— §1–§13 是 v1 设计，§14 是研究驱动的 v2 修订。下文的节号都指向它。

服务端上的一切都是手写 Python：没有 LangChain，没有 LlamaIndex，没有向量数据库。整个检索栈只有几百行可读的代码，这正是重点 —— 被评判的就是这一部分，所以它应该是可读、可做消融实验的，而不是靠配置堆出来的。

§1–§7 是服务端。§8 是客户端，延迟优化真正被听见、引用真正被看见的地方。

## 模块地图

```
backend/app/
  main.py            FastAPI 工厂 · CORS · 静态目录挂载 · 重排模型预热
  config.py          由 os.environ + 可选 .env 构造的冻结 Settings dataclass
  db.py              SQLite DDL、连接工厂、启动时迁移
  services/
    openai_client.py AIClient Protocol · OpenAIClient · FakeAIClient   ← 唯一的接缝
  ingest/
    pdf.py           PyMuPDF 几何信息（行、字号、位置），扫描件预检
    markdown.py      pymupdf4llm 文本层，由我们自己的标题驱动 → Block 流
    structure.py     五级结构阶梯 → BookStructure
    toc.py           印刷目录页解析器 · LLM TOC 层级判定
    headings.py      标题族投票（RAGFlow bullets_category）
    chunker.py       三趟 HybridChunker 移植 · 评测用的朴素基线
    enrich.py        上下文增强 · 章节／全书摘要节点
    vocab.py         采集专有名词，用于 ASR 偏置与修复
    prompts.py       入库侧的每一条 prompt，都是模块常量
    ingestor.py      后台线程编排 + 进度
  index/
    store.py         持久化 + 每本书的运行时包（LRU 3）
    dense.py         归一化 float32 矩阵上的余弦 top-k
    bm25.py          BM25Okapi + CJK／拉丁分词器
    fusion.py        加权倒数排名融合
    rerank.py        FlashRank cross-encoder，只加载一次，永不致命
  rag/
    retriever.py     混合检索 → 重排 → 父块扩展 → 上下文打包
    rewrite.py       多轮对话的查询改写
    citations.py     事后的句子 → chunk 引用归属
    pipeline.py      answer_stream()：SSE 状态机
    prompts.py       回答侧的每一条 prompt，都是模块常量
  speech/
    sentences.py     流式两阶段切分器（vendored 的 LiveKit 移植）
    pcm.py           24 kHz s16le 常量与时长计算
    turns.py         轮次标识 · 取消 · AudioPacer 背压
    phonetics.py     Double Metaphone 实体修复
  api/               books.py · asr.py · chat.py
  models.py          Pydantic 请求／响应体 —— 只在 API 边界上
  sessions.py        内存态历史，LRU 100 个会话 × 20 轮
```

---

## 1. 入库

上传落地的那一刻就在后台线程上跑起来，把进度写进 `GET /api/books/{id}/status` 轮询的那条 `books` 记录。进度条是一个 0…1 的浮点数，按阶段切分，所以它永远只向前走：

```
parsing 0.00–0.25 · structuring 0.25 · chunking 0.30
contextualizing 0.35–0.60 · summarizing 0.60–0.70 · embedding 0.70–1.00
```

被关掉的阶段就是不发生，进度条会直接跳过去 —— 没有跑过的阶段不该假装花了时间。

### 1.1 刻意的两趟提取

`ingest/pdf.py` 用 PyMuPDF 惰性遍历每一页，把每次 `get_text("dict")` 的返回压成小的 `Line` 记录（文本、字号、粗体标记、x／y），并在读下一页之前丢掉完整返回。所以一本 1000 页的书，峰值内存是平的。

这些几何记录不是被索引的东西。它们是*结构识别*要读的 —— 字号、页面顶部判定、跨页重复行检测器，都需要 markdown 已经扔掉的坐标。**被索引的文本**来自 `ingest/markdown.py`，它调用 `pymupdf4llm.to_markdown`。

为什么是两趟而不是一趟：pymupdf4llm 会还原连字，并把行末的断词连字符拼回去，而原始 `get_text()` 原样留着它们。在一本真实的 244 页书上实测，原始提取产生了 129 个连字字符和 68 处连字符换行；pymupdf4llm 产生 **0 和 0**。去连字符对 BM25 是直接的收益，因为 `light-\nhouse` 不会分词成 `lighthouse`。我们*不*从它那里拿的是它的 `IdentifyHeaders` 字体排序 —— 我们已经在同一份文档上跑过一遍五级阶梯，所以检测出的大纲通过文档记载的 `hdr_info=` 钩子回灌进去。产出的 markdown 带的是*我们的* `#` 层级，这让 `Block` 流成为我们所检测结构的一份廉价而忠实的序列化。

另外两条提取规则：

- **扫描件 PDF 预检**（§14.1）：前 10 页的每页字符数。一本原生数字书量到 ~1400+；纯图像的扫描件量到 ~0。低于 20 时上传直接失败，并给出明确的提示。OCR 不在范围内，与其悄悄产出一个空索引，不如把话说清楚。
- **页眉页脚**由跨页重复行检测器移除，而不是靠裁掉页边。裁边会顺手吃掉页眉位置偏低的那一页的正文首行，也会漏掉短页上位置偏高的页眉。

### 1.2 结构阶梯（§4.2、§14.2）

五级，按顺序尝试，每一级都比上一级更宽、也更弱：

| 级 | 信号 | 说明 |
|---|---|---|
| 1 | **内嵌 TOC** —— `doc.get_toc()` | 出版方已经把答案告诉我们了。 |
| 2 | **印刷目录页** | 在读者会称为「目录」的那几页上跑状态机：找到标记词（英文或 CJK），读取 `title … page` 条目，**用页码单调递增来校验**，再把印刷页码映射到物理页（两者差一个前置页偏移）。 |
| 3 | **标题族投票** | 移植自 RAGFlow 的 `bullets_category`。与其问「这一行看起来像标题吗」，不如让整个编号约定*族*（`Chapter <roman\|word\|number>`、`PART ONE`、`^\d{1,2}[.、 ]`、`第..[章节部]`、markdown `#`）在全书每一行上打分，取 argmax。一本书给章节编号只会用一种方式，所以胜出的族*就是*这本书的约定 —— 凡是匹配它的行都是标题，不管排版多朴素。匹配时模式从最深往浅走，所以 `1.2.3 Ferries` 是三级标题，不是一级的 `1.`。 |
| 4 | **字号直方图** | v1 的启发式：字号 ≥1.3× 正文众数字号、长度 ≤80 个字符、位于页面上方 25% 的行，按字号聚成章与节两级。 |
| 5 | **兜底** | 一个合成的「Full text」章。应用照常工作；UI 上没有 TOC。 |

**外加一个不算级的 LLM 级。** 第 1、2 级可能产出一个没有层级的*扁平*列表 —— 比如一个没有数字前缀的目录页。出现这种情况、并且没有任何离线信号（标题族）能提供层级时，用一次严格 JSON 的入库期调用给每个标题指定大纲层级，顺序不变（RAGFlow 的 `assign_toc_levels`）。它是针对一个具体缺陷的修补，不是阶梯里的一层，只要存在离线信号它就永远不会跑。

每个单元都带一个 `anchor` —— 它起始那一行的 `(page_no, y)` —— 章的 `idx` 从 1 开始，和读者看到的「第 N 章」对上。**孤立的标题永远不会自己成为一个 chunk**；它作为它下面第一个 chunk 正文的开头。

### 1.3 分块：移植 Docling 的 HybridChunker（§4.3、§14.3）

在 `Block` 流上跑三趟，叶子切分用 `semchunk`：

1. **组装。** 遍历 block，维护一个 `heading_by_level` 字典。进来的标题会弹出**层级等于或深于它自己**的每一个键 —— 一个 h2 结束上一个 h2 以及它下面的一切，但不结束它的 h1 祖先 —— 于是每个基础 chunk 都带着它被写下时所处的完整祖先面包屑。
2. **切分。** 任何*加上上下文之后* token 数超过上限的基础 chunk 都会被切开。「加上上下文之后」是承重的措辞：预算是 `available = max_tokens − header_tokens`，所以标题路径越深的 chunk 分到的正文越少，而真正被 embedding 的那份东西永远不会溢出。`semchunk` 沿着段落 → 句子 → 词的分隔符递归，在放得下的最语义化的边界上切。
3. **合并。** 相邻的过小 chunk **只在标题路径完全相同时**才合并，所以一个 chunk 永远不会横跨两个小节。最后再跑一趟切分，保证上限在合并之后依然成立。

子块是 **256–320 token，零重叠**。Chroma 的分块研究（5 个语料、472 个查询）测得递归 200/0 的 recall 是 88.1%，并且发现每一个带重叠的消融实验分组都更差：重叠买到的是重复，不是 recall。

**面包屑预算规则** —— header 是 `«Book» — Chapter — Section`，三条护栏防止它吃掉 chunk：

- header 不得超过 chunk token 预算的 **25%**；
- 正文不得低于 **96 token 的内容下限**；
- 两者无法同时满足时，路径**从头部开始裁**（先裁书名，再裁最外层），因为真正能把这个 chunk 区分开的是最具体的那一层 —— 如果连一条光秃秃的路径都放不下，那就丢掉 header，而不是饿着正文；
- **开启一个小节**的 chunk，正文里已经含着它的标题，所以它索引时完全不带 header。再重复一遍等于把同样的词索引两次。

**父块。** 一个子块的 `parent_text` 是它周围的小节文本，从子块向两侧对称生长，直到 1600 token 的上限。子块是检索*打分*的对象；父块是模型*阅读*的对象。

评测里 A 组的基线就住在同一个模块（`naive_chunks`）：直接在原始 PDF 行上切固定 800 字符的 chunk，没有 header，没有父块，页眉页脚保留 —— 所以这次消融实验量到的不只是分块的收益，还包括提取的收益。

### 1.4 增强（§14.4、§14.5）

两趟可选的 LLM 处理，都在 `AIClient` 接缝之后，都跑在一个 8 worker 的线程池里，因为它们是 IO 密集的，一本 600 chunk 的书否则会把十分钟的往返串起来一个一个做。

- **上下文增强**（`BOOKTALK_CONTEXTUAL=1`，默认开）—— Anthropic 的 contextual retrieval：由 LLM 写一句短话说明每个 chunk 处在什么位置，拼在被 embedding 和 BM25 索引的内容前面（公开的效果：检索失败率 5.7% → 2.9%）。prompt 的范围限定在 chunk 所属的**那一章**，不是整本书，这才让它负担得起。开关关掉时，静态的面包屑 header 保留 —— 那就是评测的 B 组。
- **摘要节点**（RAPTOR-lite）—— 每章一份提要，外加全书一份，作为普通的可检索 chunk 索引，但带上 `kind` 和 `level` 标记。它们回答那些扁平 chunk 已被证明答不好的主题类问题（「第 7 章讲了什么」），同时在 UI 里兼作 TOC 的提要。完整的 RAPTOR 聚类被否决了：章节树已经存在，聚类那一步是多余的。

**词表采集**在同一个阶段运行。`ingest/vocab.py` 用一个故意很笨的分布判据抽取全书的专有名词 —— 一个词必须在**不是句首**的位置以首字母大写出现至少两次，并且几乎从不以小写出现。这能留下地名 "Harbour"，丢掉副词 "However" 和普通名词 "harbour"。没有词性标注器，没有 NER 模型。CJK 被跳过：它没有大小写可读，而 Double Metaphone 是英语算法，所以一本 CJK 的书拿到的是空词表和原封不动的 v1 行为。上限 200 个词 —— 再多 ASR 的 prompt 也已经满了，而长尾正是误修复的来源。

### 1.5 Embedding 与持久化

子块按 128 一批做 embedding（被 embedding 的文档是 `header`，不是 `text`），以 float32 BLOB 存储。入库对文件 sha256 幂等：重新上传相同内容会返回已经存在的那本书 —— **200 而不是 202** —— 无论它进行到哪个阶段，所以上传一本还在索引中的书是加入正在进行的那次运行，而不是再起一次。上一次*失败*的书是例外：它的记录和它的 id 会被复用，入库重新开始。

---

## 2. 检索（§5.2 步骤 2、§14.6）

```
向量检索 top-40 ┐                    ┌ FlashRank 重排 20 个子块 ┐
                ├─ 加权 RRF (k=20) ──┤                          ├─ top-8
BM25     top-40 ┘                    └ (~8 ms, cross-encoder)   ┘
                                                  │
          ≤5 组、≤4500 tokens ── 阅读顺序 ── 父块扩展（按分数门控）
```

**两条腿都放宽。** 每条腿 40 条，不是 24 条。融合只能重排两条腿捞上来的东西，重排也只能把融合保留下来的东西排得更准。放宽的代价是一小片 numpy 和一次 BM25 argsort。

**k=20 的加权 RRF。** `score(d) = Σ_lists w / (k + rank(d))`，rank 从 1 开始。RRF 按*位置*合并，所以 BM25 的 14.2 分和 0.83 的余弦值永远不需要被换算到同一个量纲上。`k` 控制权重随 rank 衰减的快慢；常见的 k=60 来自 Cormack 等人在数百万文档规模的 TREC 语料上的工作，那里一条奖励跨列表一致性的平缓曲线是对的赌注。一本书比那小四个数量级，它的排名信息量大得多，所以适用的是小语料的建议（k=10–20）。在评测里对 {10, 20, 60} × 三种权重划分做过扫描 —— 结果是平的，所以这个默认值站在推理上，不是站在测量上。并列时按先出现的顺序决出，这让输出保持确定。

**只对子块重排。** FlashRank 的 `ms-marco-TinyBERT-L-2-v2` 在 512 token 处截断。一个 1600 token 的父块会被按它的前三分之一评判，所以只有 ~320 token 的子块参与打分。这个模块存在就是为了保证三条性质：它**永远不在 import 路径上**（两个 import 都写在 `load_reranker` 内部，所以不装它们，应用和测试套件也能启动）；它**永远不会突然下载**（`allow_download=False` 会拒绝在模型没有解包好时构造 Ranker，离线测试套件靠的就是这一点）；它**永远不致命**（每条失败路径都返回 `None`，调用方保留自己的融合顺序）。它在启动时由一个守护线程加载一次 —— 冷会话的首次推理大约是稳态的 10 倍，那不是该花在读者第一个问题上的延迟。

**带门控的父块扩展。** 一个弱匹配拖进 1600 token 的小节文本，买到的是噪声。一个 chunk 只有在最终分数落在最佳命中的 **0.3 ×** 以内时，才配得上它的父块窗口。刻意写成*比值*：这样无论 `final` 是 cross-encoder 的概率还是归一化的融合排名，它的含义都一样，而且最高分的 chunk 总会扩展，所以不会有任何问题意外地只拿到子块上下文。

**按阅读顺序装配。** chunk 按父级小节分成若干组（章节提要单独成组 —— 把摘要和它所概括的正文并在一起，会让它看起来就是那一页的内容）。各组**按最优优先**争夺 4500 token 的预算，因为预算必须保护最强的证据；然后活下来的组按 `(chapter_idx, ord)` 重新排序，以这本书自己的顺序交给模型，而不是一份按相关性打乱的顺序。组内的段落会合并并去重 —— 这既折叠掉相邻子块之间那点少量重叠，也折叠掉从同一小节切出的两个父块窗口之间大得多的重叠 —— 任何在不相邻 chunk 之间的跳跃都会标上 `[…]`，让模型不会把断口当成连续的文字。最多 5 组。

**分数是保留的，不是丢掉的。** 每个保留下来的 chunk 都带着归一化到 0..1 的 dense / BM25 / RRF / rerank 值，结果里带 `top_score`，它作为 `confidence` 出现在 `sources` 事件上。只有在 cross-encoder 真的跑过时它才是一个*绝对*量：没有 cross-encoder 时，融合排名是相对最佳命中归一化的，任何匹配上的东西都读作 1.0。

**没有任何东西以它为门。** 服务端没有拒答门控 —— 拒答是回答 prompt 里的一条规则，不是一个阈值 —— 客户端的警告也是从回答算出来的，不是从这个分数（§8.5）。这是量出来的决定，不是风格上的决定。在评测的 F 组 —— 也就是发布配置：结构感知的分块、混合检索、cross-encoder、摘要节点和上下文增强 —— `confidence` 这一列在单事实问题上平均 0.985，在被 ASR 破坏过的问题上 0.994，而在书里答不出的问题上是 0.038（D 组是 0.003）—— 但在跨章问题上是 0.469，在章节摘要问题上是 0.950，所以这种区分度在一些类别里存在，在另一些里不存在。没有重排模型时，每个类别都读作 1.000。在一本真实的书上，这个排序彻底反了：*「第 2 章发生了什么」* —— 由四段被引用的原文正确回答 —— 得分 **0.0017**，而一个书里从未提到的主题得分 **0.9953**。两个方向都错。这个数字仍然在算、仍然在发送、评测里仍然逐题上报；它只是不决定任何事。

**查询改写**（`rag/rewrite.py`）只在会话有历史时运行：一次调用，把问题加上最近 3 轮对话变成一个独立的查询。没有历史时它是恒等函数，不花任何代价。

一个**可选的单次 LLM 重排**（`BOOKTALK_RERANK=1`，SPEC §12）仍然存在，但默认关闭：它在第一个 token 之前加了一整个往返，而 cross-encoder 现在用 ~8 ms 覆盖了同样的范围。回复格式不合法时，顺序保持不变。

---

## 3. 事后引用归属（§14.7）

v1 是让模型在行文里写「(Chapter 3)」。对一个*被朗读*的产品来说这是错的工具：这个标记会被念出来，它在关键路径上消耗 token，而且模型是在猜一个它从没被展示过的归属。

所以归属发生在事后，在 `aux_pool` 上，此时 TTS 的尾巴还在合成 —— 和 RAGFlow 的 `insert_citations` 是同一个招数：

1. 把写完的回答切成**展示用句子** —— 刻意*不用*流式切分器，那个切分器很乐意在逗号处吐出一个片段。引用需要的是读者能看到一个来源挂在上面的单位，所以这个切分器只在真正的句末切（缩写表、小数点保护、CJK 句末符自带边界）。它返回的是 `(sentence, start, end)` 三元组而不是字符串；偏移量精确索引到生成出来的那份回答上，这正是客户端需要的（见下）。
2. 用**一次批量调用**把所有句子做 embedding。
3. 用一个混合相似度给每个句子对每个候选 chunk 打分：**0.9 × 余弦 + 0.1 × token 重合**。向量相似度抓住改写，而生成出来的句子大部分就是改写；token 重合抓住 embedding 抹平掉的人名、数字和日期 —— 并且被压在 0.1，永远不可能只靠虚词造出一个引用。
4. 用**递降阈值**来选：从 **0.63** 起步，每步乘 **0.8**，绝不低于 **0.3**。先要求一个有把握的匹配；只有在没有更好的时才接受较弱的。每句最多 **4 个来源**。
5. 没有任何匹配高过下限的句子**完全不带引用**。「我们不知道这句是从哪来的」是诚实的答案，客户端会把这句渲染成没有标记的样子。

候选只包括**真正到达了模型、并且已经发给客户端**的 chunk —— 被上下文预算丢掉的 chunk 两条都不满足，因为模型没读过它，客户端也没有 id 可以拿来渲染。chunk 的 embedding 直接从这本书的内存矩阵里读（已经做过 L2 归一化），所以归属永远不会重新 embedding 整本书。这个模块是纯的：一次 embedding 调用，没有数据库，没有 HTTP，没有全局状态。

这个事件在 `done` 之前紧挨着发出，名字是 `citations`：

```json
{"map": [{"sentence_idx": 0, "span": [0, 74], "chunk_ids": ["…", "…"]}],
 "turn_id": "…"}
```

**`span` 是承重的字段。** 它保存的是指向 `done` 携带的那个回答字符串的 `[start, end)` 字符偏移 —— 同一个字符串，没有 strip 过，切分器量的就是它 —— 所以客户端靠在这些偏移上*切片*回答来高亮被引用的句子。它从不重新切句。

放在客户端的切分器会是关于句子边界的第二种意见，而这份 map 只在服务端那一种意见之下才有意义。两者会恰好在困难的地方分歧：小数（`3.5`）、缩写（`e.g.`）、域名（`example.com`）、CJK 句末符。一次分歧不会大声地失败 —— 它会把后面每一个 `sentence_idx` 都错位，于是从那里开始，引用悄悄挂到错误的句子上，那比没有引用更糟。偏移量把这个问题彻底消掉。

---

## 4. 语音流水线（§5.2 步骤 4、§14.8）

延迟上的关键手法在这里。首次出声的时间主要取决于我们在第一次合成调用之前要等多少个 LLM token，所以我们诚实地等到不能再少。

### 4.1 两阶段句子切分

**一句话在哪里结束？** vendored 了一份 LiveKit Agents 基于规则的分词器（Apache-2.0，在 `speech/sentences.py` 的 vendored 代码块处署名）：一串正则把每一个不是边界的句点*标记*掉 —— `Dr.`、`e.g.`、`U.S.`、`3.5`、`example.com` —— 剩下没被标记的就是句末。只用标准库 `re`：没有 NLTK，运行时没有 `nltk.download()`，也不用随包发语料。标记用的是等长的私用区哨兵字符，所以分析副本里的每一个偏移仍然对得上原字符串。上面再拧了三道保险：

- **pipecat 的非空白前瞻** —— 看到 `.?!;…` 时，先等到下一个非空白字符再决定。他们举的例子正是要紧的那个情形：`"$29."` 不是一句话，`"$29. Next"` 是。
- **stream2sentence 的 12 字符尾部保护** —— 永远不在缓冲区最后 12 个字符里找边界，因为能消歧的那个 token 还没到。
- **绝不在数字中间切** —— 两边都是数字的句点是小数点或千位分隔符，绝不是句末。

CJK 句末符（`。！？…`）自带边界，永远是分界点。

**什么时候值得开口？** 两个阶段：

| | 触发条件 | 最小长度 |
|---|---|---|
| **第一个片段** | 任意软分隔符 —— `,;:.!?…` 及其 CJK 对应符号 | ≥ **10** 个字符，约 **30** 个词之后在下一个空格处强制发出 |
| **之后的每一块** | 只认真正的句子边界 | ≥ **20** 个字符（LiveKit 的 `min_sentence_len`），更短的向后合并 |

「In chapter seven,」完全可以拿来开口。切片带来的韵律代价止于开头，因为第二阶段从不切片。

### 4.2 线上传输的是裸 PCM

TTS 请求用 `response_format="pcm"`：无头的有符号 16 位小端单声道，24 kHz。音频事件是 `{"turn_id", "seq", "format": "pcm", "sample_rate": 24000, "b64"}`。

没有容器，没有编解码器，没有 `decodeAudioData` 这一步，两端都不需要 ffmpeg。而且因为格式固定且无头，**字节长度就是时长** —— 精确相等，不是近似 —— 背压估算靠的就是这一点。浏览器把采样直接推到同一条 AudioContext 时间线上，每个 buffer 排在 `max(currentTime, lastEnd)`，严格按 `seq` 顺序；乱序到达的会等它的前驱，永远不会被重排。

### 4.3 背压

合成按设计跑在播放前面 —— 这正是收益所在 —— 但跑得太远纯属浪费。三条边界：

- **TTS 并发 2**，队列上限是在飞行中的那些之后再排 **1** 句。
- 当 `AudioPacer` 估计已发出的音频还有超过 **5 秒**没被播放时（PCM 字节数对墙上时钟），**合成完全暂停**。
- 等待时间有一个 **30 秒的绝对上限**，因为一个估算值绝不能把一条流卡死。超时后它记日志、确认、然后继续。

结果是：一个在两句之后被取消的轮次，大约只付了两句的钱，不是六句。取消检查就放在 TTS 调用之前，所以取消到达时还排在队里的东西一分钱都不会付。而且 `finish()` 是生成器而不是列表，所以每一段合成完就立刻到达 socket —— 一次性把整条链放出来会拖慢尾巴、会在客户端还没收到的字节上就启动 pacer 的计时，还会让尾部中途的取消无事可停。

某一句的 TTS 失败只会把那一句降级成纯文本。它永远不会杀掉读者已经能看到的回答。

### 4.4 轮次生命周期、取消与历史截断

一个被朗读的回答，文本写完不算完 —— 音频被*听到*才算完。这中间的落差归 `speech/turns.py` 管。

**一个轮次的每一个 SSE 事件都带同一个 `turn_id`。** 两条不变量在 `answer_stream` 的包装层里强制，而不是在函数体内每个 `yield` 处，这样后来的改动不会漏掉哪一处：id 盖在每一个事件上，并且**轮次一旦被取消就没有事件能逃出去** —— 包括取消到达时已经在飞行中的那一段。客户端维护一个已死轮次 id 的集合，丢弃属于它们的一切。

**取消**是 `POST /api/chat/turns/{turn_id}/cancel`，带 `{played_seq}` —— 浏览器真正播完的最后一段。光关掉 socket 不够：服务端会继续合成没人会听的句子，存下来的历史会声称助手说过那些从没被说出口的话。所以客户端既 abort，也 post。

**历史截断**（vocode 的做法）是回报所在。取消时，存下来的这轮对话被改写成实际播出的那些句子，于是一个追问（「那它的港口呢？」）是在用户认得的那份记录上解析的。记成一个空回答会把这轮对话整个丢掉，当用户在一个字都还没说出口时就打断，这就是对的结果。这个接口是幂等的，并且刻意宽容：取消一个已经结束的轮次同样会修正历史，因为「用户在第二句之后就不听了」这件事，无论服务端是否已经写下第六句，都是真的。每个进程保留 64 个轮次，所以一次和流的结束赛跑的取消仍然能找到它的轮次。

### 4.5 ASR 加固（§14.9）

命名实体被破坏造成了 87–96% 的 ASR 诱发的 RAG 失败（arXiv 2608.22872），而一个被破坏的专有名词是看不见的：BM25 落空，加上一个指向别处的 embedding。一本书白送我们这个修复，因为它是一个封闭语料，里面的名字我们已经读过了。采集到的词表有两个用途：

1. **给解码器加偏置** —— 排名前 **60** 的词进入转写请求的 `prompt`，前面接一个固定开头（`"Proper nouns from the book being discussed: …"`）。Whisper 会在 224 token 处截断 prompt，所以这个列表被限制在 **896 个字符**，并按词频顺序添加：一份被截断的提示仍然带着最可能被说到的那些名字。
2. **修复仍然错回来的部分** —— 在同一份词表上做 Double Metaphone。它编码一个词*听起来*是什么样，对真正有歧义的情况返回两个编码（"Thorne" → `0RN` 和 `TRN`），于是听成第二种读法的转写仍然能匹配上。

一次错误的修复比它替掉的那次落空更糟，因为读者看到的是一个看着合理的句子，而它悄悄检索到了错的章。所以一个 token 只有在**全部**满足下列条件时才会被改写：≥4 个字符、拉丁字母、不是常见英文词、不在词表里、音标编码 ≥3 个字符（像 `FL` 这样的两字符编码会撞上半个语言）、在音标匹配之上文本相似度 ≥0.6，**并且恰好只有一个**词表词挺过了全部这些。两个候选意味着我们不知道，原词胜出。

**一个提示有两种失败方式，发布的代码两种都防了。** `prompt` 参数是建议性的，而建议性的参数会以不算错误的方式失败：

- **请求因为带了它而被拒。** `OpenAIClient.transcribe` 只捕获这一类错误，然后用同一段音频、不带 prompt 重试一次。一份名字被弄花的转写胜过没有转写。
- **回复*就是*那条 prompt。** 一个带了提示的解码器在没听到语音时，有时会把自己的提示当成转写吐回来 —— 读者录了一个问题，回来的却是服务端的名字列表。`app/api/asr.py` 用 `speech/phonetics.py:echoes_prompt` 检查每一份转写；一旦命中，同一段音频就完全不带提示重新转写一次，因为词表值得要，而问题更值得。如果重试也在回显，转写就被丢弃，接口把真正发生的事 —— 一片寂静 —— 报成 422，输入区把它渲染成「没有听清任何内容」。

**回显检测为什么匹配 prompt 开头的几个词。** `echoes_prompt` 把转写转成小写，然后问它是不是*以*模板固定开头的前 24 个字符起头 —— `"proper nouns from the bo"`。最先试的工具是转写与 prompt 之间的 token 重合率，看着更周全，却在唯一要紧的那个方向上错了。一本书的词表包含了一个短问题里的大部分实词 —— *"Who settled Kestrel Cove?"* 点到的正是提示里列出的那个地名 —— 所以真实的问题重合率很高，正在被扔掉。相比之下，那个固定开头是服务端写的文本，没有哪个读者会说出来，这让匹配上它成为证据，而不是相关性。24 个字符是权衡：长到足以没有歧义，短到能扛住解码器漏掉一个词或改掉它后面的标点。

`/api/asr` 上的 `book_id` 是可选的，未知的书会被忽略而不是被拒绝。

---

## 5. 存储

一个 SQLite 文件，`data/booktalk.db`，WAL 模式，`synchronous=NORMAL`，30 秒 busy timeout。连接按调用创建，从不跨线程共享，因为入库跑在后台线程上，同时 API 还在服务请求。

```sql
books    (id PK, title, filename, sha256 UNIQUE, pages, status, progress, error, created_at)
chapters (book_id, idx, title, page_start, page_end, PRIMARY KEY(book_id, idx))
chunks   (id PK, book_id, chapter_idx, section_title, page_start, page_end,
          text, header, parent_text, embedding BLOB, ord,
          kind TEXT NOT NULL DEFAULT 'text', level INTEGER NOT NULL DEFAULT 0)
vocab    (book_id, term, count, ord, PRIMARY KEY(book_id, term))

idx_chunks_book(book_id, ord) · idx_chunks_kind(book_id, kind)
idx_books_sha(sha256)         · idx_vocab_rank(book_id, ord)
```

`text` 是原文预览，`header` 是被 embedding 和 BM25 索引的东西（面包屑或 LLM 写的上下文 + 这个 chunk），`parent_text` 是模型读的东西。`kind`／`level` 标记摘要节点；`vocab.ord` 是排名，所以取前 N 个不用重新排序。

**启动时迁移的顺序很重要。** `init_db` 先执行建表，再做列迁移（对任何已有数据库缺少的列执行 `ALTER TABLE … ADD COLUMN`），最后建索引 —— 因为 `idx_chunks_kind` 是一个建*在迁移出来的列上*的索引，对一个在那列存在之前就创建的数据库会失败。这个顺序是一次带回归测试的 bug 修复；见 [AI_WORKFLOW.md](AI_WORKFLOW.zh-CN.md)。

**运行时包。** 对某本书的第一次查询，会把它的记录一次性装载成一个 `BookIndex`：一个归一化的 float32 numpy 矩阵、一个 BM25 索引和 chunk 元数据列表。最近使用的 3 本书的包会被缓存。在一本书的规模上（几万个 chunk），余弦 top-k 就是一次矩阵—向量乘法，远低于一毫秒，这就是这里没有向量数据库的原因。

---

## 6. AIClient 边界

```python
class AIClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def transcribe(self, audio: bytes, filename: str, prompt: str | None = None) -> str: ...
    def chat_stream(self, messages: list[dict], **kw) -> Iterator[str]: ...
    def chat(self, messages: list[dict], **kw) -> str: ...
    def tts(self, text: str) -> bytes: ...   # raw 24 kHz s16le PCM
```

应用发出的每一次网络调用都走这个 Protocol。就是这一道接缝，让整条流水线 —— 包括 HTTP 表面、SSE 流和评测框架 —— 都能离线且确定地跑起来。

- **`OpenAIClient`** 包装官方 SDK，重试 2 次，60 s 超时。`base_url` **总是显式传入**：传 `None` 时 SDK 会回退到 `OPENAI_BASE_URL` 环境变量，而 `.env` 里一行空的 `OPENAI_BASE_URL=`（dotenv 会把它读成 `""`）就会悄悄弄坏每一个请求。`config.py` 把空值规范成 `None`，客户端再替换成真正的默认值。当配置的模型在某个账号上不可用时，ASR 和 TTS 回退到 `whisper-1` / `tts-1`。
- **`FakeAIClient`** 返回由哈希做种的确定性 embedding（`sha1(text)` → `RandomState` → 单位向量）、逐句合成的假 PCM、一份固定的转写和固定的对话回复。调用方用 `purpose="rewrite" | "rerank" | "answer" | "toc_levels" | "context" | "summary"` 给补全打标；`OpenAIClient` 会剥掉这个标记，而 `FakeAIClient` 用它返回符合那个调用点*形状*的回复 —— `toc_levels` 拿到一个 JSON 层级列表，`context` 拿到一句定位句，依此类推。相同的文本永远得到相同的向量，但*相似*的文本得到的是不相关的向量：测试只能依赖精确文本的确定性，绝不能依赖语义相似度。

`create_app(settings=..., ai=..., store=...)` 三个都收，所以整个 HTTP 表面可以对着假客户端跑，不需要任何 monkeypatch。完全没有 key 时，`build_client` 什么都不返回，依赖模型的接口回 503 并附上说明，而不是以晦涩的方式失败。

---

## 7. 线上协议

```
GET    /api/health                     → {ok, openai_configured}
POST   /api/books  (multipart "file")  → 202 {book_id, status}，新内容
                                       | 200 {book_id, status}，该 sha256 已入库
                                         或正在入库（§1.5）
GET    /api/books                      → {books:[{book_id, title, pages, status,
                                                  progress, created_at}]}
GET    /api/books/{id}/status          → {status, progress, error?}
GET    /api/books/{id}/toc             → {title, chapters:[{idx, title, page_start,
                                                  page_end, synopsis?, opening?}]}
DELETE /api/books/{id}                 → 204
POST   /api/asr    (multipart "audio"，可选 "book_id")
                                       → {text, duration_ms}
POST   /api/chat   {book_id, session_id, question}
                                       → SSE
POST   /api/chat/turns/{turn_id}/cancel  {played_seq}
                                       → {turn_id, cancelled, played_seq, answer}
```

对于两者都没有的章节，`synopsis` 和 `opening` 是**缺失，而不是 null**：响应用 `response_model_exclude_unset` 序列化，所以客户端只需要检查 `chapter.synopsis` 这一件事。`/api/asr` 上的 `duration_ms` 是转写花了多长时间 —— 客户端已经知道自己那段录音的长度，而服务端不解容器就量不出音频时长。

SSE 是在 `StreamingResponse` 上手写的 —— 线上格式是四行代码，比 `sse-starlette` 少一个依赖。响应头带 `X-Accel-Buffering: no`，这样没有代理会压着这条流。一轮的事件顺序：

```
status(rewriting|retrieving) → question → [status(retrieving)] → sources
→ status(generating) → delta* → status(speaking) → audio(seq 0..n)* → citations → done
```

| 事件 | 载荷 |
|---|---|
| `status` | `{"stage": "rewriting" \| "retrieving" \| "generating" \| "speaking"}` |
| `question` | `{"text"}` —— 检索实际跑的那个查询，也就是有历史可改写时的改写结果，否则是原样的问题 |
| `sources` | `{"sources": [{"chunk_id", "chapter_idx", "chapter_title", "section_title", "pages": [start, end], "preview", "text", "scores": {"dense", "bm25", "rrf", "rerank", "final"}}], "confidence"}` |
| `delta` | `{"text"}` —— 回答的一个 token 级切片，边流边出 |
| `audio` | `{"seq", "format": "pcm", "sample_rate": 24000, "b64"}` |
| `citations` | `{"map": [{"sentence_idx", "span": [start, end], "chunk_ids"}]}`（§3） |
| `done` | `{"answer", "timings": {"rewrite_ms", "retrieve_ms", "first_token_ms", "first_audio_ms", "total_ms"}}` |
| `error` | `{"message"}` —— 失败时取代后面的整个尾部 |

`preview` 是 `text` 的前 200 个字符；`scores` 是 §2 里那些归一化的信号，`confidence` 是其中最好的 `final`。每个事件都带 `turn_id`，在一处统一盖上（§4.4），而不是在每个 `yield` 处。

---

## 8. 客户端

`frontend/src` 是 Vite 上的 React 18，用 TypeScript 写：约 4,100 行 TSX 和约 3,500 行 CSS。没有状态库，没有路由，没有组件框架。留在这里的依赖都在边界上挣到了自己的位置：`eventsource-parser` 负责 SSE 分帧，三个 Radix 原语（tooltip、popover、collapsible）负责没人应该重写一遍的焦点与键盘行为，`use-stick-to-bottom` 负责对话线程的滚动锚定，三个 `@fontsource-variable` 字体让排版不成为运行时的网络依赖。

```
frontend/src/
  main.tsx           React 根 · 三个字体 · 三份样式表
  App.tsx            两个视图、共享的 AudioQueue、飞行用的矩形、toast
  hooks/
    useChat.ts       一条实时 SSE 轮次 → 按书分的对话线程 · 按轮次 id 取消
    useLibrary.ts    书目列表 · 上传 · 入库期间 1 s 轮询状态 · TOC 缓存
  lib/
    api.ts           带类型的 fetch 包装 + ApiError · 发后不管的 cancelTurn
    sse.ts           基于 POST 的 SSE 读取器（eventsource-parser + AbortController）
    audioQueue.ts    base64 PCM → 一条 AudioContext 时间线，无缝，严格按 seq
    recorder.ts      MediaRecorder 按住说话；对外暴露输入节点供电平表使用
    types.ts         §7 的线上契约，用 TypeScript 写
  components/
    Shelf.tsx        落地页：刻度导轨、书行、滚动恢复、帮助
    ShelfBook.tsx    CSS 3D 里的一册书 · 打开的动作 · 飞回书架
    ShelfRail.tsx    左边缘固定的刻度索引
    Reader.tsx       打开的一本书：顶栏、书本面板、对话栏
    BookPanel.tsx    飞行的封面 · 从它里面托出的章节页 · 翻页器
    Conversation.tsx 对话线程 · 一轮问答 · 按 span 切片的回答 · 来源面板
    SourceCard.tsx   一条页边注；它的出处把书翻到那段原文
    Composer.tsx     麦克风和文本框，以及 ASR 的那次往返
    MicButton.tsx    录音控件和它的实时电平环
    Visualizer.tsx   一块 canvas 波形，可以指向任一方向的音频
    Hero.tsx         空书库面板，两个界面共用
    IngestPanel.tsx  一本书正在索引时的六个入库阶段
    StatTile.tsx     唯一的数字展示面：那个百分比，逐步累加
    Toast.tsx · Tip.tsx · Icons.tsx   通知、提示条、图标集
    cloth.ts         每本书的配色：书架位置 → 五个自定义属性
  styles/
    tokens.css       设计 token —— 一个亮色世界，一个主题色
    base.css         reset、排版、跟随主题的浏览器表面、减弱动效
    components.css   每个组件的规则
```

### 8.1 两个界面，没有路由

`App.tsx` 在一个 `useState` 里存 `view: "shelf" | "chat"`，这就是全部的导航模型。代码里有三件事，在路由之下都得想办法绕开：

- **凡是必须活过这次穿越的东西，都住在两个界面之上。** 一个 `AudioQueue`（`useMemo`，只构造一次）、对话线程、书库和它的轮询，全都挂在 `App` 上。一次会卸载阅读页的路由切换，会让正在朗读的回答哑掉，并把一条活着的 SSE 轮次扔在地上。在这里，离开阅读页会停掉这一轮，是因为 `backToShelf` 调了 `stopAll()` —— 这是一个决定，不是一次卸载。
- **离开不是瞬间完成的。** 返回时，书架分支*同时*在自己上方渲染正在离场的 `Reader` 300 ms（`leavingReader`），这样两个界面可以一起动（§8.2）。两个界面同时挂载，用一条路由很难表达。
- **转场是作为量出来的几何数据传过去的。** 这段飞行由一个 `DOMRect` 驱动，它作为 prop 从一个界面交到另一个界面。URL 装不下它，而一个深链接到达时也没有可以起飞的地方。

代价是实在的，值得说明：没有可分享的书籍链接，浏览器的后退按钮会离开应用，而不是回到书架。对两个界面来说这是划算的；第三个界面会改变这个答案。

**书架**是落地页，也是唯一一个在书库为空时还能立住的界面。**阅读页**是一本书 —— 书本身在左，对话在右，输入区停靠在它下面。它的主栏由这本书的状态决定：书库为空时是 `Hero`，书正在索引或者已经失败时是 `IngestPanel`，就绪之后是 `Conversation`，默认是一个空的 `.studio__body`，让输入区在过渡帧里保持贴底而不是跳动。正在离场的那份阅读页拿到 `flightFrom={null}`、空操作的回调和一个空的主体：那一刻它只是布景，它原本捧着的那个对象已经在书架上了。

两道保险把这个闭环收好。阅读页开着时删掉最后一本书，会把视图送回书架 —— 已经没有什么可待在里面了。而 `activeBookId` 变化会执行 `stopAll()` *并且*清掉打开的引用，因为章节号跨书没有意义：留着它，会把下一本书翻到一个不相干的章。

**状态归属**按生命周期划分，不按界面：

| Hook | 拥有 | 说明 |
|---|---|---|
| `useLibrary` | `books`、`loaded`、`activeBookId`、`tocs` | 只要有书在入库，就每 1 s 轮询 `/status`，某本书转为就绪时重新拉一次列表。它的 `INGESTING` 集合镜像后端的非终态状态 —— 漏掉一个就会在入库中途悄悄停掉轮询。TOC 拉取失败会被吞掉：TOC 是增强，书照样能回答。 |
| `useChat` | `threads`（按书）、`streamingBookId`、`speaking` | 同时只有一轮在跑（`activeRef`）、一个已死轮次 id 的集合，以及每本书在第一次提问时铸出、并在整个页面生命周期里保持的一个 session id —— 于是服务端的历史按书隔离，切换书不会拿另一本书的记录去解析追问。 |
| `App` | `view`、`flightFrom`、`returnFlight`、`leavingReader`、`cited`、toast | 一切关于*这次穿越*、而不是关于某一本书的东西。 |

Toast 按消息去重，所以一个不断失败的轮询没法把同一句话叠六层；启动时的一次健康探测会把「没有 OpenAI key 也在运行」的提示抬到输入区上方，而不是让第一个问题去发现它。

### 8.2 一个对象穿越过去

打开一本书是一个对象的一段连续运动。两个界面上的其他一切都给它让路，然后再回来。

**出去，书架 → 阅读页。**

1. 点击给那一行设上 `opening`。CSS 把这册书从它的书架姿态里取出来 —— 46° 的倾斜变平，上升 16px，在 150 ms 内放大到 1.08 —— 三张书页从封面下面散开，彼此错开 40 ms。
2. **280 ms** 之后，`ShelfBook` 量 `.shelf-book__volume` 并把矩形往上交。测量刻意放在散开*之后*：飞行必须从书现在所在的位置起飞，而不是从它打开之前站的位置。
3. `App.openBook` 记下书架的 `scrollTop`，选中这本书，把矩形存成 `flightFrom`，然后切换视图。
4. `BookPanel` 在 `useLayoutEffect` 里对着那个矩形跑一次 FLIP：面板渲染在它该在的位置，封面从书原来的位置动画进来。

真正在动的是一个节点 —— `.tome__block`，那张封面，带着书名。封面把文字一起带过去，因为没有它，书到达时是空白的，书名在交接处消失，那看起来像剪辑而不是移动。字标完全不做动画：两个界面在同样的内缩位置画同一个 `.brand` 块，所以它是连续的，而不是过渡出来的。背景也是同一个元素 —— `App` 在两个分支之外渲染 `.app-ground`（三片缓慢的色场），只改 `--ground-strength`，书架上是 1，阅读页里在 900 ms 内变成 0.28，于是穿越是注意力的转移，而不是场所的更换。

动画内部有三个决定，每一个都是一次修复：

- **用 Web Animations API，不用 CSS transition。** 在 `requestAnimationFrame` 里释放的 transition，只要 React 在飞行中途重渲染就会丢掉这次释放。一个 `Animation` 对象扛得住，而且可以被取消。
- **动的是盒子，不是 transform。** 书在书架上是一根宽条，在阅读页里是一张竖立的封面，所以这段飞行改变的是它的*形状*，不只是位置。缩放会把书名一起拉变形；动画化 `width`／`height` 让书名能随着封面转正而重新排版。
- **测量之前先取消正在跑的动画。** React 在开发模式下会把 effect 挂两次，而一个残留的飞行会把元素停在它的第一帧上 —— 在那里它的矩形是书架的，不是目的地的。

**着陆在哪里被保证。** `land()` 在任何可能失败的东西*之前*就被安排好，并且靠一个 `done` 标记做到幂等。它可以从四个方向到达：动画的 `finished` promise、减弱动效的分支、没有节点或者目的地量到 0×0 时的提前返回，以及一个 `FLIGHT_MS + 200` = **820 ms** 的兜底定时器。着陆才是打开这本书的动作：它设上 `landed`，露出章节页并加上 `.tome--open`，并调用 `onArrived` 让 `App` 能清掉 `flightFrom`。跑不起来的飞行只是一次糟糕的转场；打不开的页面是一个坏掉的界面，所以先上定时器，动画是锦上添花。

章节页被压到着陆之后才画，理由是相关的：它是飞行元素的子节点，太早画会让一张半透明的章节页飘过对话区。

**回来，阅读页 → 书架。** `backToShelf` 停掉这一轮，量下 `.tome__block` 此刻站的位置，存成 `{bookId, rect}`，然后*立刻*切换视图 —— 接着把阅读页作为一个固定、不接受指针事件的图层，继续挂在书架上方 300 ms。这段重叠正是重点：书开始变回它那一行的同时，阅读页的各个面板正在离场，而不是等它们走完之后。

`ShelfBook` 跑反向 FLIP，比去程多一步：它**把那一行放平了量**。一册书静止时的矩形，是一个倾斜 46° 的物体的包围盒，对一本 148px 高的书来说那是 103px 高；动画停在这个数上，书看起来明显被压扁了。所以 transform 会为了正好一次测量而摘掉，然后立刻装回去，飞行结束在真实的盒子上，倾斜和侧倾都复原。在这之前它先调 `onBeforeMeasure()`，因为书架必须先滚回阅读页离开时的位置，某一行才能说出自己在哪 —— 这也是为什么 `Shelf` 把滚动恢复保存成一个只跑一次的*函数*，而不是放在滚动容器的 ref 回调里：返回的那一行的 layout effect 先跑。

**减弱动效**是在每一道接缝上分别兑现的，而不是全局压掉：散开的延迟塌成 0，两段飞行都直接短路到着陆，书页、界面的进出场、被引用页的光环和缓慢漂移的背景全部停下，导轨的 `scrollIntoView` 从 `smooth` 换成 `auto`。

### 8.3 音频通路

```
POST /api/chat ─▶ eventsource-parser ─▶ useChat.onEvent ─▶ audioQueue.enqueue
                                                                 │
                                        base64 → Int16LE → Float32│
                                                                 ▼
                                AudioBuffer 起播于 max(currentTime, lastEnd)
```

**SSE 走 POST。** 浏览器的 `EventSource` 只能 GET，既带不了 JSON body 也不能被 abort，所以 `lib/sse.ts` 用 POST 发出去，把 `response.body` 通过 `TextDecoderStream` 灌进 `eventsource-parser` 的 `EventSourceParserStream`。库负责线上语法 —— 被切在两个数据块之间的事件、多行 `data:`、注释 —— 这个模块负责属于我们的那两部分：POST 契约，以及一个取消读取器的 `AbortSignal`，保证 abort 之后不再有事件被投递。非 2xx 永远不会变成一条流：它抛出一个带着服务端自己的 `detail` 的 `ApiError`，toast 显示的就是它。

**轮次身份先看。** 在按事件名分发之前，`useChat` 先读 `turn_id`：如果它在已死集合里，事件就被丢掉，这正好是读者已经打断掉的那一轮的迟到音频段。第一次看到的 id 会存到这一轮上，并补到这条对话记录上，好让取消接口有东西可以指名。

**PCM，手工解码。** `decode()` 做 base64 解码，把字节按有符号 16 位小端采样走进一个 `Float32Array`（`/ 32768`；末尾多出的单个字节被忽略，而不是当成半个采样），再拷进 `ctx.createBuffer(1, n, 24000)`。没有 `decodeAudioData`，没有容器，没有编解码器。不是 `"pcm"` 的格式回退到 `decodeAudioData`，而解码抛异常则返回 `null` 并被*跳过*，而不是让它把队列卡住。

**无缝排期。** `drain()` 只放行 `seq` 恰好轮到的那一段，把它排在 `max(ctx.currentTime, nextStartTime)`，并按这个 buffer 自己的时长推进 `nextStartTime` —— 于是连续的段在同一条时间线上严丝合缝地相接，乱序到达的会在 `pending` 里等它的前驱，而不是被重新排序。`currentTime` 是第一段、以及任何一次欠载之后的下限，这让队列不会把音频排进过去。在 `stop()` 之后才完成的解码会被一个 `epoch` 计数器丢弃：它属于一个已死的轮次。

**`playedSeq` 的意思是听到了，不是发出去了。** `maxPlayedSeq` 唯一的写入者是某个 source node 的 `onended`，而 `stop()` 会在停掉每个节点之前把它的 `onended` 置空 —— 所以能走到那个回调，只可能意味着这一段自然播到了尾。服务端截断存储历史用的就是这个数字（§4.4），所以它必须精确地是这个意思。

**空洞在结束时放行。** 收到 `done` 时客户端调 `finish()`：不会再有缺失的 `seq` 到达了，所以卡在空洞后面的一切都会播出来。某一句的 TTS 失败时后端会跳过一个 `seq`（§4.3），为这个空洞干等，会让整段回答的其余部分因为丢了一句而全部哑掉。

**取消，按这个顺序。** `stopAll()`：

1. 把活着的 `turn_id` 加进已死集合，于是已经在飞行中的东西都被忽略；
2. 读 `audioQueue.playedSeq` 并带着它 POST `/api/chat/turns/{id}/cancel` —— 发后不管，`keepalive: true`，错误吞掉，因为取消绝不能挡住下一个问题；
3. abort 掉 fetch；
4. 把这轮对话终结为 `stopped`，保留屏幕上已经有的那部分文本；
5. 停掉每一个已排期的节点，清空 `pending`，把顺序重置回 seq 0。

`playedSeq` 是服务端用来改写历史的唯一那个数字，所以它在这次拆除的两端都被保护：`useChat` 在第 5 步之前读它，而 `stop()` 自己在重置之前把 `maxPlayedSeq` *冻结*进 `frozenPlayedSeq`（把已播的秒数冻进 `frozenPlayed`），于是在队列拆完之后再读它的 getter 仍然答得对。

`ask()` 先调 `stopAll()`，所以问一个新问题也就是打断旧问题的方式。`audioQueue.start()` 在点击或提交的处理函数里同步调用 —— Safari 和 iOS 只允许在用户手势里创建或恢复 `AudioContext` —— `Composer` 在麦克风按下时通过 `primeAudio` 做同样的事，在任何 await 之前。

**一块波形，两个方向。** `Visualizer` 接一个 context 和一个节点，画的是真实的 analyser 数据；录音时输入区把它指向录音器的 `MediaStreamAudioSourceNode`，朗读时回答卡片把它指向 `audioQueue.outputNode` —— 播放信号经过的那个 gain node。描边用元素自己的 `color`，所以一个 class 就决定方向：实时输入是珊瑚色，播放是蓝色。

### 8.4 引用通路

客户端是 §3 的渲染器，它的设计出自一条规则：**服务端的句子边界是唯一的边界**。

**用切片来高亮。** `FinalAnswer` 拿 `done` 里的 `answer` 字符串和 `citations` 里的 `map`，保留 `span` 格式正确且在范围内的条目（`0 ≤ start < end ≤ answer.length`，至少一个 chunk id），按起始偏移排序，然后用一个游标走这个字符串：span 之前的文本原样输出，span 本身变成一个 `.sentence` 元素，最后一个 span 之后的尾巴原样输出。与前一个重叠的 span 会被跳过而不是嵌套，所以文字永远不会重复或丢失。没有条目的句子保持无标记 —— 这是「我们不知道这句是从哪来的」的诚实呈现。

客户端里任何地方都没有句子切分器，这是刻意的。它会成为关于句子在哪里结束的第二种意见，在小数和缩写上和服务端分歧，而一次分歧就会悄悄把后面每一个 `sentence_idx` 挪到错误的文本上。

**流式过程中**，回答是累积起来的 `delta` 片段，不带高亮：span 这时还不可能存在，因为归属是在写完的回答上跑的。收到 `done` 时这些片段被丢掉，服务端的 `answer` 成为偏移量所索引的那唯一一个字符串。

**这条链接是双向的**，靠每轮对话一份状态（`hoveredChunk`）：悬停一个被引用的句子，会点亮它指名的那些来源卡片；悬停一张卡片，会点亮每一个引用它的句子 —— 句子上是一层 `--blue-soft` 的底色，卡片上是一圈内嵌的蓝色描边加 1px 的抬升。

**来源面板**是一个 Radix `Collapsible`，标题写着「Used N sources」，里面装着同一份列表的两种呈现：

- **页边注**（`SourceCard`）—— 回答右侧页边上的白色卡片，每张带它的编号、`Ch. N · pp. a–b` 这样的出处、章与节的标题，以及原文本身，截断到四行。只有在这张卡片确实藏了东西时，「Full passage」开关才会出现：原文超出了它 200 字符的 `preview`，或者那个 preview 长到会被截断裁掉；
- **一份紧凑的轨迹** —— 每个来源一行：序号、出处、章与节、chunk id。

哪一种可见，由**对话栏上的容器查询**决定，而不是视口：`.reader__talk` 是一个 inline-size 容器，在*栏宽*低于 980px 时页边被去掉，同样那些 `SourceCard` 内嵌进折叠面板里（视口 1280px 处有一个做同样切换的兜底）。阅读页把屏幕的 53% 给了对话，所以一个以视口为准的断点，会把页边注印进一个根本没有页边可印的栏里。

**把书翻到被引用的原文。** 卡片上的出处是一个按钮。按下它会把 `{chapterIdx, pages, text, chunkId}` 提到 `App` 成为 `cited`；`BookPanel` 找到 `idx` 是那个值的章并翻过去 —— 但印出来的是**这段原文自己的页码和自己的文字**，不是这一章的开头。「它出自的那一章」不是答案出自的地方，而一个三十页长的章的开头，会让读者去找他刚刚听到的那句话。这一页以 chunk id 为 key，所以第二次引用会重放从书里托出来的那个动作，而不是原地把文字交叉淡入；它到达时带着一圈蓝色光环，在 1.4 s 内落定 —— 长到足够找到它，短到不至于一直在那儿发亮 —— 而这段原文带着和回答里那个被引用句子相同的底色，于是这条链接的两端看得出是同一个颜色。移动翻页器会把引用清回这一章的开头；切换书也一样。

### 8.5 溯源警告是在回答上量出来的

回答上方的那条警告 —— *"None of this answer could be traced to a passage — check it against the book"* —— **不是**由检索的置信分数驱动的。它在三件事同时成立时出现：这一轮已经结束、`citations` 到了并且是**空的**、回答长度超过 **80 个字符**。

这个条件是一个计数：回答自己的句子里，有多少条能被事后归属追回到一段检索到的原文。一条都没有，而回答又长到已经说了点什么，这才是值得标出来的情况 —— 而且这是对回答的判断，不是对查询的判断。为什么要在那里量，这件事本身也是量出来的，在一本真实的书上：*「第 2 章发生了什么」*由四段被引用的原文正确回答，检索置信度却是 **0.0017**，而一个书里从未提到的主题得分 **0.9953**。两个方向都错，所以这个分数扛不起这件事。引用覆盖率把同样这两种情况干净地分开了，因为它是从读者被要求信任的那个东西上算出来的。

80 字符的下限是这条规则的另一半：一次拒答（「书里没有讲到这个」）本来就很短，而且*按设计*不引用任何东西，为它报警是没道理的。

这个分数仍然在算、仍然随 `sources` 事件发送、评测里仍然作为 `confidence` 逐题上报；`useChat` 会把它解析到这轮对话上。回答卡片里没有任何东西以它为条件。

### 8.6 样式系统

三份样式表，由 `main.tsx` 按顺序加载：`tokens.css`（词汇表）、`base.css`（reset、排版、浏览器表面）、`components.css`（所有规则）。没有 CSS-in-JS，没有原子类框架。JavaScript 唯一写进去的样式，是那些*值*本身就是数据的自定义属性：一本书的布面、麦克风的实时电平、背景的强度。

**Token。** 刻意只有一个亮色世界 —— 系统里任何地方都不存在深色表面，所以没有第二套色板需要维持诚实。一块画布（`#e9ebf2`），在 `<body>` 上带两片静态的径向晕染，白色卡片，一个明确的主题色（`--blue`），以及只留给录音、别处不用的珊瑚色（`--live`）。刻度是固定的而不是流体的：八级字号从 12 到 44，一套 4px 的间距刻度，三种卡片阴影，都以同一个头顶光源按偏移 + 模糊构造，另外还有两种给书架上的书册用 —— 它们是*放在*地面上而不是浮在其上，所以在投影之外还要一层接触阴影。每一组配色搭配旁边的注释里都写着实测的对比度，这让这套色板是可核对的，而不只是被声称的。

**玻璃，只用在三个表面上。** toast、书架的帮助按钮和章节翻页器做磨砂 —— 这三个是真正浮在滚动内容之上的表面，规则就是这一条。每一个都声明两次，顺序很重要：*默认*那次声明是不透明的 `--glass-solid`（0.92 白），带 `backdrop-filter` 的 0.72 半透明版本只在 `@supports (backdrop-filter: blur(1px))` 里叠上去。所以一个没有这个滤镜的浏览器得到的是一块不透明的面板，而不是浮在移动内容上的文字，两种状态都量过 —— 这也是 `tokens.css` 为每一种都记了一个对比度的原因。那两片环境晕染就是为这个而存在的：玻璃底下需要颜色，否则它只是在灰上磨出灰。

**每本书的布面。** `cloth.ts` 是一本书颜色的唯一来源，而且它是这本书*在书库中位置*的纯函数。`clothForIndex(index)` 返回五个自定义属性 —— 两个渐变端点、页缘的染色、布面上的墨色，以及渐变的角度 —— 书架的行和阅读页的封面都读它，所以一本书在两个地方是同一个颜色。正是这一点，让一个对象能在两个界面之间穿越而不用换装。

色板是 16 组配好的色相对 × 2 个调子 = **32 种布面**。在色相环上均分 32 步的做法试过，错了两次：在某一个饱和度上它读起来像一套荧光笔，而两个端点只差几度时，封面根本看不出渐变。这些色相对每一组跨越 24–70° 的色相，并且是作为组合挑出来的 —— 桃色到玫瑰、薄荷到天空、长春花到兰紫 —— 第二个轴是调子：同一组的一个轻盈版和一个更深的版本。*按位置*走这套色板而不是对书名做哈希，才是让一面书架看起来是被编排过的原因：前十六本书拿到那十六组色相对，接下来十六本用更深的调子重复它们，只有第三十三本书才会穿上书架上已经出现过的颜色。哈希对谁都不保证这一点 —— 两本书可能在第二行就撞色。渐变角度也在走（`112 + (slot % 5) × 14`），所以相邻的书不只是色相不同。这个模块记下了约束这一切的对比度：`--ink` 在全部三十二种的每一个端点上都量到 ≥7.7:1，布面上带染色的次级墨色在最差的一种上是 4.89:1。

**动效是状态，不是装饰。** 界面反馈是 120–240 ms，共用一条缓动曲线；那些长数字（620 ms 的飞行、900 ms 的背景）属于那唯一一段*本身就是主体*的运动，而不是关于主体的反馈。`base.css` 在 `prefers-reduced-motion` 下把 animation 和 transition 的时长压掉，而每一段由 JS 驱动的运动自己也检查同一个查询 —— `element.animate()` 不是 CSS 动画，那条一刀切的规则够不着它。

---

## 9. 借来的算法

对着我们自己的数据结构重新实现，从不复制粘贴，只有明确 vendored 的那一块除外。完整的调研和被否决的方案：[RESEARCH.md](RESEARCH.zh-CN.md)。

| 借用 | 来源 | 落在哪里 |
|---|---|---|
| `bullets_category` 标题族投票 | RAGFlow | `ingest/headings.py` |
| `assign_toc_levels` 严格 JSON 层级判定 | RAGFlow | `ingest/toc.py` |
| `insert_citations` 混合句子归属 | RAGFlow | `rag/citations.py` |
| `HybridChunker` 三趟组装 | Docling（IBM，MIT） | `ingest/chunker.py` |
| chunk 大小／零重叠的证据 | Chroma 分块技术报告 | `ingest/chunker.py` |
| contextual retrieval | Anthropic | `ingest/enrich.py` |
| RAPTOR 在章节层级上的退化情形 | RAPTOR（ICLR 2024） | `ingest/enrich.py` |
| `split_sentences` 规则分词器（**vendored**，Apache-2.0） | LiveKit Agents | `speech/sentences.py` |
| 非空白前瞻 | pipecat | `speech/sentences.py` |
| `minimum_first_fragment_length`、`force_first_fragment_after_words`、`context_size` | stream2sentence | `speech/sentences.py` |
| `SentenceStreamPacer` 的背压形态 | LiveKit Agents / RealtimeTTS | `speech/turns.py` |
| 带生成 id 的取消、PCM 流播放器 | openai-realtime-console | `speech/turns.py`、`frontend/src/lib/audioQueue.ts` |
| 把历史截断到实际播出的部分 | vocode | `rag/pipeline.py` |
| 回答可能没有依据时的诚实 UI 警告 | kotaemon | `Conversation.tsx` —— 借来的是这个*做法*，信号不是：kotaemon 用检索分数报警，而那个分数在这里两个方向都测错了（§2），所以这条横幅改成数被追溯到的句子（§8.5） |
| 评测框架的形态（分类、逐查询 CSV） | Onyx `run_search_eval.py` | `eval/run_eval.py` |
