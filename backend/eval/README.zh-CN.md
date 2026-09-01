# BookTalk 评测工具

[English](README.md) · **中文**

`app/` 里的这条流水线，检索效果真的比朴素分块更好吗？其中又是哪一部分在起作用？本目录用一本没有人读过的测试用书、88 道由写书所依据的同一张事实表推导出来的问题，以及六个彼此之间恰好只差一处的消融实验分组来回答这两个问题（SPEC §10、§14.10）。

```
facts.py               编造的事实表 —— 12 座岛 × 8 条事实
make_fixture_book.py   渲染 PDF 并生成 golden.json
run_eval.py            运行各分组，写出 report.md 和几份 CSV
fixture/               生成的 PDF（重新生成，不要手改）
golden.json            生成的问题集（重新生成，不要手改）
report.md              最近一次运行的表格
results/               per_question.csv · per_category.csv · sweep.csv
```

## 运行方式

```bash
cd backend
.venv/bin/python eval/make_fixture_book.py     # ~2 s, writes the PDF + golden.json
.venv/bin/python eval/run_eval.py --fake       # ~6 s, offline, no API key
.venv/bin/python eval/run_eval.py --fake --sweep
```

真实模型的运行会在运行时从环境变量（或 `backend/.env`）读取 `OPENAI_API_KEY`，并且从不打印它：

```bash
.venv/bin/python eval/run_eval.py              # all six arms, real models
.venv/bin/python eval/run_eval.py --arms A,F   # just the two endpoints
.venv/bin/python eval/run_eval.py --sweep-only --no-answers
```

常用参数：`--limit N`（每个类别只取 N 道题，用于冒烟测试）、`--no-answers`（只算检索指标 —— 不产生生成的花费）、`--reuse`（沿用上一次运行留在 `eval/.work` 里的索引，不重新构建）、`--data-dir`、`--report`、`--results`。

一次真实运行会在这本 25 页的书上构建四套索引，并向六个分组各提 88 道问题：在 `gpt-4o-mini` / `text-embedding-3-small` 上大约 500 次对话补全和 800 次 embedding 调用，外加为 F 组的索引给每个 chunk 做一次上下文增强调用。`--no-answers` 能去掉其中约四分之三。

## 测试用书

*A Field Guide to the Meridian Archipelago* —— 25 页，12 章，每章两节，带内嵌大纲、书眉和页码。书里的一切都是编造的，这正是重点：没有模型读到过 Kestrel Cove，所以答对说明检索找到了那段文字，而不是说明模型本来就知道。每章把八条结构化事实（创建者、特许年份、人口、地标及其高度、当地菜肴、到首都的距离、节庆及月份、出口物产）编织进普通叙述文字里，周围还环绕着关于同一座岛、容易误命中的相近文字。

`make_fixture_book.py` 不会写出一份它无法担保的测试用书。在生成任何东西之前，它会检查：每个预期答案子串都确实印在书里；每一句证据句都能原样通过 PDF 抽取；成书的 PDF 经过应用自己的结构识别重新解析后得到十二个命名正确的章节；以及全部十二处 ASR 讹误都能用 `app.ingest.vocab` 从同一批文字中提取出的词表修复。

## 问题集

| 类别 | n | 考察什么 |
|---|---|---|
| `single-fact` | 48 | 一条事实、一章、一句证据。 |
| `cross-chapter` | 8 | 需要两章都被检索到的比较题。 |
| `chapter-summary` | 12 | “第 7 章讲的是什么” —— 主题性问题，扁平 chunk 答得差，§14.5 的摘要节点答得好。 |
| `not-in-book` | 8 | 听起来合理、但书里没有答案的问题。正确行为是拒答。 |
| `asr-corrupted` | 12 | 岛名被听错的事实题（“Castrel Cove”），在检索前用书里的词表修复（§14.9）。 |

`golden.json` 里的每一条都带有 `question`、`expected_answer_substrings`、`chapter_idx`、`gold_chapters`（cross-chapter 类别为两章）、`expect_refusal` 和 `evidence` —— evidence 是一个完美的检索器会返回的那些原句，也是 token-IoU 的评分对象。`run_eval.py` 在加载时会重新校验整个文件，所以被手工改过的标准答案集不会悄悄产出没有意义的数字。

## 各分组

| 分组 | 索引 | 检索 |
|---|---|---|
| A | 固定 800 字符的 chunk，不带标题，不做父块扩展 | 仅向量检索 |
| B | 结构感知的 chunk，带面包屑标题 | 仅向量检索 |
| C | 同 B | 向量检索 + BM25，加权 RRF |
| D | 同 B | + cross-encoder 重排 |
| E | + 章级摘要节点和全书摘要节点 | 同 D |
| F | + 每个 chunk 的 LLM 上下文增强 | 同 D |

每次运行构建四套索引（A、B/C/D、E、F 之间能共用的就共用），每个分组都完全按服务端的调用方式走 `ingest_book`、`retrieve` 和 `build_messages`。检索深度取 10 而不是生产环境的 8，这样 recall@10 才有定义；配置的其余部分都没有改动。

## 指标

- **recall@{1,3,5,10}** —— 一道问题的*标准章节*出现在前 k 条结果中的比例。按标准章节取平均、而不是只问“有没有命中其中任意一章”，才让 cross-chapter 这一类的分数诚实：两章里找到一章是一半成功，不是完全成功。
- **MRR** —— 第一条来自任一标准章节的 chunk 的排名倒数。
- **token-IoU** —— Chroma 提出的精确率信号：标准证据句与实际展示给模型的拼装上下文之间，按 token 多重集算的交并比。这个数值天然就小（一句话对上 4500 token 的预算），只有分组之间的比较才有意义。
- **answer match / answer full** —— 生成的答案中出现的预期子串比例，以及所有预期子串都出现的问题占比。
- **refusal ok** —— 双向计分。既要对书外问题拒答，*也要*对其余问题不拒答；否则一条对什么都拒答的流水线看起来会是满分。
- **confidence** —— `Retrieval.top_score`。只有在重排模型跑过时它才是一个绝对量。这里只做测量，没有任何东西以它为门槛：正是这套评测显示它在真实的书上会反过来（一道关于第 2 章、答对了的问题得 0.0017，而书里从未提到的主题得 0.9953），所以最终上线的依据警告改为统计被追溯到出处的句子数，也没有做拒答闸门。
- **repair rate** —— ASR 讹误被完全还原的比例。

除此之外还有一份首次命中的排名分布，因为 MRR 相同的两个分组，分布形状可能差别很大；以及 `--sweep` 在 RRF `k ∈ {10, 20, 60}` 和向量检索/BM25 权重划分上的网格搜索（TREC 的 k=60 假设的是百万篇文档的语料，一本书比它小四个数量级，所以这里是实测而不是照搬）。

## `--fake` 能证明什么，不能证明什么

`FakeAIClient` 用 `sha1(text)` 作为 embedding 的种子，因此相似的文本会得到毫不相关的向量：**向量检索在构造上就是噪声**，只用向量检索的 A、B 两组分数接近零，高于它们的部分全靠 BM25 撑起来。它的对话补全是一句写死的话，所以每一项答案指标衡量的都是这个假客户端。D 组的 cross-encoder 被换成一个确定性的词法替代实现，因为 FlashRank 需要下载模型。

一次通过的 `--fake` 运行能证明：评测工具、四套索引配置、六套检索配置、ASR 修复路径和报告写入器都能端到端跑通。值得引用的数字来自带真实 key 的运行，`report.md` 会在自己的表头里写明产生它的是哪一种运行。

## 结果

本目录下的 `report.md` 保存着最近一次运行 —— 目前是一次真实模型运行（六个分组加上 RRF 网格搜索，88 道问题；表头里写明了所用的模型）。同一张表和分类别的分析也出现在项目 README 中；`results/` 保存它背后的 per-question、per-category 和 sweep CSV。
