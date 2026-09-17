# Engineering explanation

**Project:** RAG Question Answering (local FastAPI + FAISS + Gemini)

## 1. Design parameter and why it was selected

`RELEVANCE_THRESHOLD = 0.45` (cosine similarity after L2-normalized MiniLM vectors). It is a configurable constant in `app/config.py`, not a magic number scattered through the code. It was chosen as an empirical starting point after measuring scores on the sample leave policy: the in-document question *“How many annual leave days do employees receive?”* scored **0.768**, while *“What is the company's maternity leave policy?”* scored **0.3637** and *“What is the capital of France?”* scored **−0.013**. A cut at 0.45 therefore kept the supported question and dropped the unsupported ones **without calling Gemini**. This is not a universally optimal threshold; it should be retuned on a labelled evaluation set.

## 2. Real failure observed during development

Live generation against `GEMINI_MODEL=gemini-2.0-flash` failed with `google.genai.errors.ClientError: 404 NOT_FOUND` — *“This model models/gemini-2.0-flash is no longer available. Please update your code to use models/gemini-3.6-flash.”* **Cause:** the default model id in `.env.example` was stale relative to the current Gemini API. **Fix:** default and `.env.example` were changed to `gemini-3.6-flash`, and 404s are mapped to a generic “model was not found” API error that does not echo secrets. **Lesson:** pin and re-verify model ids at integration time; do not assume a flash alias stays valid.

## 3. Metric tracked and what it told us

End-to-end query latency is returned as `latency_ms` and logged as embedding, retrieval, LLM, and total. On 17 Sep 2026, after uploading the sample TXT, a **known** question took **3423.8 ms** (Gemini dominated; a scripted LLM-only timing was ~3095 ms). The **maternity** question took **18.8 ms** with `llm_latency_ms = 0` because retrieval filtering skipped the model. The metric showed that skipping the LLM on insufficient evidence is both a correctness control and a large latency win.

## 4. What was not finished and what would be next

There is no labelled retrieval/generation evaluation set, so the 0.45 threshold and 800/120 chunk settings remain starting values. Next: a small gold set (in-doc / out-of-doc / paraphrase), precision@k and refusal accuracy, then a threshold sweep. Also missing: document delete/rebuild, per-user indexes, and a recorded walkthrough video (placeholder only in the README).
