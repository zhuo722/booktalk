# 开源方案调研

[English](RESEARCH.md) · **中文**

设计定稿之前，五个并行的调研 agent 分别负责一个领域，把 GitHub 和相关文献过了一遍 ——
要求是读真实源码、逐条核实结论（python 版本支持、依赖体积、延迟），而不是凭记忆作答。
其中几项建议在调研过程中就直接在目标机器上做了基准测试（FlashRank、pymupdf4llm、
chonkie+model2vec）。完整的建议排序见
[research/raw-recommendations.txt](research/raw-recommendations.txt)。最终采纳的集合写在
SPEC.md §14 里；本文件记录的是每个想法的来源，以及哪些是有意舍弃的。

## 采纳了什么，来自哪里

| 采纳项 | 来源 | 起决定作用的证据 |
|---|---|---|
| 事后做句子→chunk 的引用归属 | RAGFlow `insert_citations()` | 朗读出来的文本里不出现任何标记；约 60 行代码加一次 embedding 批量调用，与 TTS 并发执行 → 首次出声延迟不增加 |
| 结构识别用标题体例投票兜底 | RAGFlow `bullets_category` | 纯正则在 5 种标题体例之间投票；让结构识别在没有内嵌 TOC 的 PDF 上仍然能用 |
| 用 LLM 判定 TOC 层级（入库时） | RAGFlow `assign_toc_levels` prompt | 严格 JSON 的一次性调用；解决 TOC 没有数字前缀的书 |
| 章节摘要索引节点（RAPTOR 的简化版） | RAPTOR（ICLR 2024），经由 RAGFlow | 章节树本来就有；约 15 次 LLM 调用就能拿到 RAPTOR 在主题类问题上的大部分收益，不需要 UMAP/GMM 聚类 |
| pymupdf4llm 抽取层 | pymupdf4llm | 在一本真实的 244 页书上实测：原始文本有 129 个连字字符 / 68 处连字符断行，处理后是 0 / 0 —— 去掉连字符断行直接带来 BM25 recall 的提升；能装在 py3.9 上 |
| 三遍式 chunk 组装 | Docling `HybridChunker`（IBM，MIT 协议） | 有名字、可引用的算法：用 `heading_by_level` 生成标题路径，按上下文算 token 预算，只在标题路径完全相同的范围内合并；`semchunk` 叶子切分器在 py3.9 上验证通过 |
| 子块约 256–320 token，零重叠 | Chroma 分块技术报告 | 5 个语料 / 472 条查询：recursive 200/0 的 recall 是 88.1%；重叠有可测量的负面影响 |
| RRF 之后接 FlashRank 重排 | FlashRank（TinyBERT-L-2-v2） | 在本机实测：模型 5 MB，单独跑一次 `rank()` 处理 20 段文本耗时 7.6 ms，不依赖 torch，有 cp39 wheel —— 这项技术在 Anthropic 的测量中把失败率从 2.9% 降到 1.9%。在评测里*每条查询*的开销要更大：平均检索耗时从 187 ms（C 组）变成 267 ms（D 组），这约 80 ms 是整个检索调用的增量，而不只是 cross-encoder 本身 |
| 对每个 chunk 做 LLM 上下文增强 | Anthropic contextual retrieval | 公开数据是检索失败率从 5.7% 降到 2.9%；按章节为单位处理，把入库成本压在合理范围内；这一项由我们自己的消融实验验证，而不是直接假定成立 |
| RRF 常数与权重调优 | Elasticsearch/OpenSearch 文档、Anthropic cookbook | k=60 来自百万级文档的 TREC 语料；小语料的建议值是 k=10–20；Anthropic 的 cookbook 自己用的就是加权融合（0.8/0.2） |
| TTS 分句的两段式策略 | stream2sentence + LiveKit Agents | 第一个片段在软分隔符处切开，长度 ≥10 字符（首次出声提前约 0.4–0.8 s）；后面的 chunk 要求真正的句子边界且 ≥20 字符 |
| 把分句器源码内置进仓库 | LiveKit `_basic_sent`（Apache-2.0）+ pipecat 的前瞻逻辑 | 只用标准库 `re` 就能处理 Dr./Ph.D/3.5/URL；运行时不需要下载 NLTK 数据 |
| PCM 流式播放 | openai-realtime-console `WavStreamPlayer` | 通过 SSE 传原始 24 kHz s16le：播放无间隙，不用解码 mp3，整条链路不出现 ffmpeg，并且能按采样点精确统计“已经听到了多少” |
| 带生成 id 的取消 + 历史截断 | openai-realtime-console + vocode | 迟到的 chunk 按失效的 turn_id 丢弃；存下来的历史截断到真正播放过的句子 |
| TTS 背压（队列上限，未播放内容超过 5 s 就暂停） | LiveKit `SentenceStreamPacer`、RealtimeTTS | 一次被取消的轮次只浪费一句话的 TTS 花费，而不是整段回答 |
| ASR 实体偏置 + Double Metaphone 纠正 | arXiv 2608.22872 | 命名实体被识别错，占 ASR 引发的 RAG 失败的 87–96%；书本身是封闭词表，这个修复几乎不花成本 |
| 如实呈现检索情况的界面（低置信度提示条） | kotaemon | 采纳的是“如实告知”这个界面模式，kotaemon 的*机制*没有沿用。它的提示条由归一化的检索得分驱动，而在一本真实的书上我们这边的得分是反的 —— “第二章讲了什么”被正确回答时得分 0.0017，书里根本没提过的主题得分 0.9953。最终上线的提示条统计的是：回答里有多少句能被事后引用归属追溯到某个段落，当一段有实质内容的回答（>80 字符）一句都追溯不到时才显示 |
| 来源分成“已引用”和“同时检索到”两组 | Onyx | 按 chunk id 去重，已引用的排在前面 |
| 评测框架的组织方式（分类、每条查询一行的 CSV） | Onyx `run_search_eval.py` | 分类对应真实的失败模式，包含拒答和 ASR 噪声这两个分组 |
| 用 Radix 基础组件，不用组件框架 | 前端调研结论 | 无样式、零依赖地拿到交互行为（焦点锁定、portal、aria），同时避开一眼能认出的 shadcn 外观 |
| eventsource-parser / use-stick-to-bottom / Literata | 前端调研 | 每一项都替换掉一处手写代码，而聊天界面正是在这些地方会明显出问题；全部是零依赖或接近零依赖，已在 React 18 上验证 |
| Tufte 边注式引用版式 + 纸质感配色 | Tufte CSS、Readwise Reader、tweakcn vintage-paper、Midday | 在约 65ch 行宽下做边注，既是可选方案里最有辨识度的版式，也把检索质量直接展示了出来 |

## 舍弃了什么，为什么

| 舍弃项 | 原因 |
|---|---|
| LangChain / LlamaIndex | 检索是这次评分的核心；这类抽象会把我们需要展示和测试的决策藏起来 |
| ragas / deepeval | 两者在 py3.9 上都能装上，但会拉进很重的依赖树（langchain/datasets/scipy；grpcio/otel/posthog 遥测）；LLM 评判分数不适合用来分辨消融实验之间的细小差异。实际做出来的只有确定性指标，没有别的：recall@{1,3,5,10}、MRR、token 级 IoU、预期子串匹配、拒答正确性。SPEC §14.10 里原本计划和这些指标一起写的那个约 30 行的手写忠实度评判器一直没有写 —— `backend/eval/` 里没有任何评判器 |
| Late chunking（Jina） | 在这套架构里根本做不了：它需要 token 级的 embedding，而 OpenAI 只返回池化后的向量；公开的收益也只有 +1.5–1.9 nDCG |
| HyDE | 在首次出声的关键路径上加一次完整的 LLM 往返；而且会把实体幻觉出来 —— 这正是语音链路最承受不起的一类失败 |
| 完整的 RAPTOR | 为 2–3 个点的收益付出 UMAP/numba 的入库成本，而且和一棵真实的 TOC 树大量重复；改为采纳它的退化情形，即章节摘要 |
| 把语义分块作为主路径 | NAACL 2025（Qu、Tu & Bao）和 Chroma 都发现，在真实文档上它相对固定长度分块的成本不划算；只保留为一个可选的消融实验分组 |
| 用 Docling / marker / MinerU 做解析器 | 都是 torch 量级的模型栈；对一个跑在 py3.9、只用 CPU 的应用来说体量不对，何况 pymupdf4llm 加自己写的分级结构识别实测结果已经很干净 |
| Tailwind / shadcn / assistant-ui / AI Elements | shadcn 的默认外观是最容易被认出来的 AI 模板风格 —— 评分标准里明确会扣分；Radix 基础组件能给出同样的行为而不带这套外观 |
| 用 wavesurfer.js 做播放可视化 | 它需要一个可解码、且预先算好波峰的 blob；按句子流水线推送的 PCM 队列永远拿不到这样的数据 —— 一个约 40 行的 analyser 加 canvas 组件就能同时覆盖麦克风和播放 |
| 现在就上浏览器端 VAD（@ricky0123/vad-web） | 首次加载约 4.8 MB，仓库里有 iOS 相关和打断可靠性的 issue；默认用按住说话，时间允许的话之后再加 RMS 端点检测 |

## 方法说明

调研由五个并行的 agent 完成（RAG 产品 / PDF 解析 / 语音链路 / 检索与评测 / 前端），消耗约
500k token、309 次工具调用、约 16 分钟墙钟时间。每个 agent 返回结构化的结论，并对每个仓库
给出 `concerns` 字段；留下来的建议被写进 SPEC.md §14 和前端构建说明。star 数和维护状态都
对着 GitHub 实时数据核对过（其中：Verba 已归档，chatbot-ui 和 vocode 大约 2 年没有更新 ——
受欢迎程度没有被当成适用性的依据）。
