# DocuQA: Document-Grounded Question Answering with RAG

A Retrieval-Augmented Generation (RAG) assistant that answers questions about the
B.M.S. College of Engineering Academic Rules and Regulations and cites the source
page for every answer. It compares four retrieval strategies and includes a
refusal guard that says "Not found in the document" for out-of-scope questions.

Course project: Natural Language Processing for LLM (24AM7PCLLM), Department of
Machine Learning, B.M.S. College of Engineering, Bengaluru.

## Features

- Upload a text-based PDF and ask questions in plain English
- Answer, retrieval confidence, response time and cited source pages
- Four retrieval methods compared: BM25, dense, hybrid, hybrid + cross-encoder reranker
- Chunk-size ablation: 128, 256 and 512 words
- Refusal guard tuned with a similarity threshold (0.65)
- Web interface built with Gradio

## Pipeline

PDF -> text extraction -> chunking (20% overlap) -> BM25 index + BGE embeddings

Question -> hybrid retrieval (top 20) -> cross-encoder rerank (top 3) -> Flan-T5-large -> answer + cited pages

If the best similarity is below 0.65, the system replies "Not found in the document."

## Models

| Component | Model |
|---|---|
| Sparse retrieval | BM25 (rank_bm25) |
| Dense retrieval | BAAI/bge-small-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generator | google/flan-t5-large |

All models are pre-trained and used without fine-tuning.

## Results

Evaluated on 40 answerable questions (49 questions in total including 9 out-of-scope).

| Chunk size | Method | Recall@1 | Recall@5 | MRR | Latency (ms) |
|---|---|---|---|---|---|
| 128 | BM25 | 0.525 | 0.750 | 0.624 | 1.05 |
| 128 | Dense | 0.525 | 0.825 | 0.649 | 12.09 |
| 128 | Hybrid | 0.575 | 0.925 | 0.695 | 11.88 |
| 128 | Hybrid + Reranker | 0.725 | 0.900 | 0.807 | 60.19 |
| 256 | Hybrid | 0.575 | 0.925 | 0.731 | 12.45 |
| 512 | Hybrid | 0.625 | 0.925 | 0.756 | 12.20 |

Full results are in `results.csv`.

Key findings:

- Best configuration: hybrid + reranker with 128-word chunks (MRR 0.807)
- Hybrid retrieval beats BM25 and dense alone at 128 and 256 words
- The reranker helps at small chunks but costs 5 to 12 times more latency
- Refusal guard at threshold 0.65: 88.9% of out-of-scope questions refused,
  2.5% of valid questions wrongly refused (see `threshold_table.csv`)

## Run it

1. Open `docuqa_rag.ipynb` in Google Colab and select a GPU runtime
   (Runtime -> Change runtime type -> T4 GPU).
2. Upload your PDF as `syllabus.pdf` and your test set as `qa.csv`.
3. Run the cells in order. The last cell launches the Gradio demo and prints a public link.

Or locally:

```
pip install -r requirements.txt
```

Then run the notebook with Jupyter.

## Test set format (qa.csv)

| Column | Meaning |
|---|---|
| question | The question text |
| snippet | A short exact phrase from the PDF that contains the answer (empty for out-of-scope) |
| answerable | 1 if the PDF contains the answer, 0 otherwise |

## Repository structure

```
docuqa-rag/
  docuqa_rag.ipynb      main notebook (evaluation + demo)
  qa.csv                test set (49 questions)
  results.csv           retrieval results
  threshold_table.csv   refusal threshold trade-off
  requirements.txt
  screenshots/          demo screenshots
  README.md
```

## Limitations

- Small test set (49 questions) and a single document
- The document contains both older and amended rules, so an older clause can be retrieved
- The refusal threshold was tuned on the same questions used for evaluation
- Text-based PDFs and English questions only; tables are flattened into text
- The small generator can misread correct context (Flan-T5-base answered one question wrongly; Flan-T5-large fixed it)

## Future work

- LoRA fine-tuned reranker or embedder
- Larger test set and faithfulness metrics for generated answers
- Version-aware retrieval that prefers amended clauses
- Kannada and Hindi queries with a multilingual embedding model

## Team

- [Roshanth.V] ([1BM23AI155])
- [SS Gokula] ([1BM23AI158])

Faculty in-charge: Upeksha Prasad G, Department of Machine Learning, BMSCE

## Acknowledgements

Built with Hugging Face Transformers, Sentence-Transformers, rank_bm25 and Gradio.
