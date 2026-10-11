# RAG evaluation plan

This project deliberately does not claim a made-up accuracy or latency number. Before making an accuracy claim in a resume, build and run an evaluation set.

## Create a small gold set

Choose at least 8–10 videos you are permitted to analyse. For each video, write a small set of questions from the transcript and record the expected supporting time range. Include:
- answerable factual questions,
- questions that need details spread across more than one passage,
- paraphrased questions that do not reuse the transcript's exact wording,
- unanswerable questions whose answer is not in the transcript,
- adversarial transcript content that tries to instruct the model to ignore its system prompt.

Keep the source transcript and expected evidence manually checked. Do not use model-generated answers as the only ground truth.

## Metrics

- **Recall@k:** proportion of questions for which at least one gold evidence chunk appears in the top k retrieved chunks.
- **MRR:** average reciprocal rank of the first gold evidence chunk.
- **Citation support rate:** proportion of factual answer claims whose cited sources support the claim, judged by a human or a separately validated rubric.
- **Abstention quality:** whether questions with no transcript evidence are correctly declined, and whether answerable questions are not declined unnecessarily.
- **Latency:** measure transcript fetch, embedding/index build, retrieval and generation separately; report median and p95 after repeated runs.
- **Cost:** record embedding and answer-model usage per indexed video and per question using provider usage data where available.

## Repeatable protocol

1. Freeze the test questions and expected evidence before changing retrieval settings.
2. Record model IDs, chunk size, overlap, top-k, test date, and dependency versions.
3. Run the same dataset before and after a change.
4. Review failures by type: retrieval miss, bad chunk boundary, ambiguous question, unsupported generation, missing citation, or unavailable captions.
5. Report dataset size and measurement method with every metric. Never generalize from a tiny demo set to all YouTube videos.

## Initial baseline

The application currently uses overlapping timestamp-aware caption chunks, FAISS similarity search with top-k = 4, and Gemini for answer generation. No measured retrieval or answer-quality score is published yet.


## Manual-transcript mode evaluation

Include the same gold questions in both automatic-caption and pasted-transcript modes. Test both inline timestamped captions and plain text without timestamps. Verify that a timestamped source link points to the expected moment, and that an un-timestamped source is labelled as having no timestamp rather than receiving a misleading 00:00 citation.
