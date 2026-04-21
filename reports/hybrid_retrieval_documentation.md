# Hybrid Retrieval + Reranker 文档

## 1. 目标

本次交付对应 `improve.md` 中的第 2 项：补 `Hybrid Retrieval + Reranker`。  
原来的历史检索链路只有一条 dense 向量检索路径，虽然已经支持：

- `summary + transcript chunks` 双写
- `meeting_id` 过滤
- `chunk_type` 过滤

但本质上仍然是 `dense-only retrieval`。这会导致两个问题：

1. 对关键词、术语、短语型查询的命中不稳定
2. top-k 排名完全依赖 embedding，相近语义的噪声结果可能排得过高

这次改动的目标不是重做整套 RAG，而是在现有 `Chroma + sentence-transformers` 的基础上，把检索入口升级成一个更合理的默认策略。

## 2. 当前实现

核心代码在 [src/meeting_ai/retrieval.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/src/meeting_ai/retrieval.py)。

现在 `MeetingVectorStore.query()` 默认执行四步：

1. `dense retrieval`
2. `BM25 lexical retrieval`
3. `RRF merge`
4. `CrossEncoder rerank`

也就是说，默认查询路径已经从：

`dense-only`

升级为：

`hybrid(dense + lexical) -> reranker`

## 3. 技术选型

### 3.1 Dense 检索

继续复用原有向量链路：

- 向量库：`ChromaDB`
- embedding：`sentence-transformers`
- 默认模型：`intfloat/multilingual-e5-small`

这样不会破坏现有的数据写入逻辑，也不用迁移索引格式。

### 3.2 Lexical 检索

新增：

- `rank_bm25`

实现方式不是单独维护第二套数据库，而是在查询时：

- 先按 `meeting_id` / `chunk_type` 过滤候选文档
- 再在过滤后的文档上做一次内存级 BM25 打分

这样做的优点是工程改动小，而且和现有 metadata filter 天然兼容。

### 3.3 Hybrid Merge

Dense 分数和 BM25 分数的数值分布不同，直接线性加权容易不稳。  
所以这里没有做 score normalization，而是用了：

- `RRF (Reciprocal Rank Fusion)`

RRF 的好处是：

- 不需要把 dense score 和 lexical score 拉到同一量纲
- 对小规模候选集很稳
- 工程实现简单

### 3.4 Reranker

新增：

- `sentence_transformers.CrossEncoder`
- 默认模型：`BAAI/bge-reranker-base`

Reranker 只在 merge 后的候选池上做打分，不会对全集做全量 pair scoring。  
这样成本可控，也更符合“先召回、后精排”的常见工业链路。

## 4. 查询流程

### 4.1 Dense 路径

先对 query 编码，然后从 Chroma 取 top-k dense candidates。

### 4.2 Lexical 路径

再对过滤后的文档集合做 BM25 打分。  
中文 tokenization 采用轻量实现：

- 英文/数字按词切
- 中文按单字切

这个方案不是分词最优解，但足以支撑当前会议检索 MVP。

### 4.3 Merge

将 dense candidate list 和 lexical candidate list 做 union，再用 RRF 合并排序，得到 `hybrid_seed_score`。

### 4.4 Rerank

如果 `RETRIEVAL_ENABLE_RERANKER=true`，对 merge 后的候选池调用 CrossEncoder，生成 `reranker_score`，最终返回 top-k。

如果 reranker 失败：

- 不中断查询
- 直接回退到 merge 后的候选排序
- 在返回 metadata 中附带 `reranker_error`

## 5. 配置项

新增环境变量：

```env
RETRIEVAL_STRATEGY=hybrid
RETRIEVAL_DENSE_CANDIDATE_K=12
RETRIEVAL_SPARSE_CANDIDATE_K=12
RETRIEVAL_RRF_K=60
RETRIEVAL_ENABLE_RERANKER=true
RETRIEVAL_RERANKER_MODEL=BAAI/bge-reranker-base
RETRIEVAL_RERANKER_CANDIDATE_K=8
```

默认建议保持当前值即可。

## 6. 返回结果变化

`RetrievalRecord.metadata` 现在会附带更多调试信息，例如：

- `retrieval_strategy`
- `dense_score`
- `lexical_score`
- `hybrid_seed_score`
- `reranker_score`
- `retrieval_sources`

这对调试为什么某条文档排在前面很有帮助。

## 7. 测试覆盖

新增/升级的测试在 [tests/test_retrieval.py](D:/Autumn%20Campus%20Recruitmen/Meeting%20AI/tests/test_retrieval.py)：

- 写入 `summary + transcript chunks`
- hybrid 默认路径
- reranker 分数透传
- `meeting_id` / `chunk_type` 过滤
- `dense` 路径回退

推荐验证命令：

```powershell
conda activate meeting-ai-w1
python -m pytest tests/test_retrieval.py tests/test_orchestrator.py tests/test_ui_app.py -q
```

## 8. 当前限制

### 8.1 BM25 仍是内存级重建

当前 lexical index 是在查询时对过滤后的文档集合动态构建。  
对于现在这个项目规模完全够用，但如果文档量继续变大，就应该升级为真正的 sparse index。

### 8.2 还没有 retrieval benchmark 数据集

这次改动已经把工程链路补齐了，但还没有正式的 retrieval eval manifest。  
所以现在可以明确声称：

- 已实现 hybrid retrieval + reranker
- 默认检索链路已升级

但还不应该编造 `Recall@5` 或 `MRR` 这种实验数字。

### 8.3 中文 tokenization 是轻量方案

当前为了避免引入额外中文分词依赖，中文 lexical tokenization 用的是单字切分。  
这足够支撑 MVP，但如果后续要把中文关键词检索做得更稳，可以再补：

- `jieba`
- 自定义术语词典
- 会议领域专用词表

## 9. 结论

这次交付把项目的会议记忆检索从：

- `vector-only prototype`

推进到了：

- `hybrid retrieval prototype with reranking`

虽然它还不是大规模工业级检索系统，但已经补上了一个非常关键的能力缺口，也让这个项目在面试和答辩里更接近真实 RAG 工程实践。
