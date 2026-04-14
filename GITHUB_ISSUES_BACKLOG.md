# GitHub Issues-Ready Backlog

Use this file to create GitHub issues directly. Each section includes a suggested title, labels, assignee role, milestone, and checklist.

## Suggested labels to create first
- `epic`
- `p0`, `p1`, `p2`
- `ml`, `mle`, `backend`, `qa`, `data`
- `evaluation`, `inference`, `training`, `security`, `reliability`
- `needs-spec`, `blocked`, `good-first-task`

---

## Milestones
- `M1: Foundation (Weeks 1-4)`
- `M2: Detection + Recognizer (Weeks 5-10)`
- `M3: Serving + Flywheel (Weeks 11-16)`

---

## EPIC: P0 Data Foundation & Evaluation Baseline
**Title:** `[EPIC][P0] Data foundation, writer-aware splits, and CER/WER baseline`  
**Labels:** `epic`, `p0`, `data`, `evaluation`, `ml`  
**Assignee:** `@ml-lead`  
**Milestone:** `M1: Foundation (Weeks 1-4)`

### Checklist
- [ ] Dataset manifest schema and validator implemented.
- [ ] Writer-aware split generator with deterministic seed.
- [ ] CER/WER evaluator and baseline report committed.
- [ ] CI gates for data validation and metric reporting added.

### Child issues
1. `[P0][DATA] Define dataset manifest schema + validator`
2. `[P0][ML] Build writer-aware split generator`
3. `[P0][EVAL] Add CER/WER evaluator + baseline benchmark report`
4. `[P0][MLE] Add CI checks for data integrity and baseline metrics`

---

## Issue 1
**Title:** `[P0][DATA] Define dataset manifest schema + validator`  
**Labels:** `p0`, `data`, `ml`, `needs-spec`  
**Assignee:** `@mle-engineer`  
**Milestone:** `M1: Foundation (Weeks 1-4)`

**Description**
Create a dataset contract and a validation script for OCR training/evaluation data.

**Acceptance criteria**
- [ ] Schema includes `sample_id`, `writer_id`, `split`, `line_transcript`, `image_uri`, quality tags.
- [ ] Validator detects malformed rows, missing files, empty transcripts, and invalid split values.
- [ ] Validation summary reports counts by split and writer.
- [ ] Exit code is non-zero on validation failure.

**Tasks**
- [ ] Add `schemas/manifest.schema.json` (or equivalent).
- [ ] Implement `scripts/validate_manifest.py`.
- [ ] Add sample valid/invalid manifest fixtures for tests.
- [ ] Add usage docs in README/developer docs.

---

## Issue 2
**Title:** `[P0][ML] Build writer-aware split generator`  
**Labels:** `p0`, `ml`, `data`  
**Assignee:** `@ml-engineer-1`  
**Milestone:** `M1: Foundation (Weeks 1-4)`

**Description**
Generate deterministic train/val/test splits by writer to avoid leakage.

**Acceptance criteria**
- [ ] No writer appears in multiple splits.
- [ ] Script supports configurable split ratios and random seed.
- [ ] Split stats exported (samples, writers, avg transcript length).
- [ ] Script is reproducible with the same seed.

**Tasks**
- [ ] Implement `scripts/make_writer_splits.py`.
- [ ] Add unit tests for split integrity.
- [ ] Export `splits/train.csv`, `splits/val.csv`, `splits/test.csv`.

---

## Issue 3
**Title:** `[P0][EVAL] Add CER/WER evaluator + baseline benchmark report`  
**Labels:** `p0`, `evaluation`, `ml`  
**Assignee:** `@ml-engineer-2`  
**Milestone:** `M1: Foundation (Weeks 1-4)`

**Description**
Create standardized OCR evaluation scripts and baseline report.

**Acceptance criteria**
- [ ] Evaluator computes CER and WER.
- [ ] Supports slice metrics (length bucket, quality tag, source).
- [ ] Baseline report generated in markdown.
- [ ] Results can be compared across model versions.

**Tasks**
- [ ] Implement `evaluate.py` with CER/WER utilities.
- [ ] Add benchmark template `reports/baseline.md`.
- [ ] Add CLI for `--predictions` and `--references` input files.

---

## Issue 4
**Title:** `[P0][MLE] Add CI checks for data integrity and baseline metrics`  
**Labels:** `p0`, `mle`, `evaluation`  
**Assignee:** `@mle-engineer`  
**Milestone:** `M1: Foundation (Weeks 1-4)`

**Description**
Integrate data validation and baseline evaluation checks into CI.

**Acceptance criteria**
- [ ] CI runs manifest validator on PRs.
- [ ] CI verifies split integrity script passes.
- [ ] CI uploads evaluation artifact summary.

**Tasks**
- [ ] Add CI workflow job.
- [ ] Add failure annotations for invalid manifests.
- [ ] Persist artifact report for reviewers.

---

## EPIC: P0 Line Detection Pipeline
**Title:** `[EPIC][P0] Add line detection pipeline for document OCR`  
**Labels:** `epic`, `p0`, `ml`, `inference`  
**Assignee:** `@ml-lead`  
**Milestone:** `M2: Detection + Recognizer (Weeks 5-10)`

### Checklist
- [ ] Detector integrated with pluggable backend (DBNet/CRAFT).
- [ ] Cropping pipeline outputs line images + metadata.
- [ ] Detection quality harness with precision/recall/IoU.

### Child issues
5. `[P0][ML] Integrate DBNet/CRAFT line detector`
6. `[P0][QA] Build detection quality harness`

---

## Issue 5
**Title:** `[P0][ML] Integrate DBNet/CRAFT line detector`  
**Labels:** `p0`, `ml`, `inference`  
**Assignee:** `@ml-engineer-1`  
**Milestone:** `M2: Detection + Recognizer (Weeks 5-10)`

**Acceptance criteria**
- [ ] Detector returns line polygons/boxes + confidence.
- [ ] Crops saved with traceable IDs and source linkage.
- [ ] Config toggles backend and threshold values.

**Tasks**
- [ ] Implement `detect.py` abstraction.
- [ ] Add backend adapters for DBNet and/or CRAFT.
- [ ] Add crop export format for recognizer ingestion.

---

## Issue 6
**Title:** `[P0][QA] Build detection quality harness`  
**Labels:** `p0`, `qa`, `evaluation`, `ml`  
**Assignee:** `@qa-engineer`  
**Milestone:** `M2: Detection + Recognizer (Weeks 5-10)`

**Acceptance criteria**
- [ ] Labeled subset exists for detection validation.
- [ ] Precision/recall/IoU reported by threshold.
- [ ] Benchmark report stored and versioned.

**Tasks**
- [ ] Create annotation subset and data card.
- [ ] Implement evaluation notebook/script.
- [ ] Add threshold sweep report.

---

## EPIC: P0 Sequence Recognizer Upgrade
**Title:** `[EPIC][P0] Upgrade to line-level recognizer (CRNN+CTC + LM)`  
**Labels:** `epic`, `p0`, `ml`, `training`, `inference`  
**Assignee:** `@ml-lead`  
**Milestone:** `M2: Detection + Recognizer (Weeks 5-10)`

### Checklist
- [ ] CRNN+CTC training and inference baseline complete.
- [ ] Beam decoder and LM rescoring integrated.
- [ ] CER/WER improvement report produced.

### Child issues
7. `[P0][ML] Implement CRNN+CTC recognizer baseline`
8. `[P0][ML] Add language model rescoring for decoding`

---

## Issue 7
**Title:** `[P0][ML] Implement CRNN+CTC recognizer baseline`  
**Labels:** `p0`, `ml`, `training`  
**Assignee:** `@ml-engineer-2`  
**Milestone:** `M2: Detection + Recognizer (Weeks 5-10)`

**Acceptance criteria**
- [ ] Trains end-to-end on line crops.
- [ ] Supports variable-length transcripts.
- [ ] Outputs CER/WER and saved checkpoints.

**Tasks**
- [ ] Add `recognize.py` model + trainer.
- [ ] Add augment config for line-level handwriting.
- [ ] Add inference script for batched line inputs.

---

## Issue 8
**Title:** `[P0][ML] Add language model rescoring for decoding`  
**Labels:** `p0`, `ml`, `inference`, `evaluation`  
**Assignee:** `@ml-engineer-1`  
**Milestone:** `M2: Detection + Recognizer (Weeks 5-10)`

**Acceptance criteria**
- [ ] Decoder supports score fusion with tunable weights.
- [ ] Ablation report compares greedy/beam/beam+LM.
- [ ] Domain lexicon support is optional via config.

**Tasks**
- [ ] Implement `decode.py` with LM fusion.
- [ ] Add config keys for alpha/beta and beam width.
- [ ] Add evaluation script for decoding ablations.

---

## EPIC: P1 Serving & Runtime Optimization
**Title:** `[EPIC][P1] Build OCR inference service and optimize runtime`  
**Labels:** `epic`, `p1`, `backend`, `mle`, `inference`  
**Assignee:** `@backend-lead`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

### Checklist
- [ ] FastAPI service with health/detect/recognize endpoints.
- [ ] Structured output schema with confidences and bboxes.
- [ ] Runtime benchmarks and ONNX path documented.

### Child issues
9. `[P1][BACKEND] Implement FastAPI OCR service`
10. `[P1][MLE] Add ONNX export + runtime benchmark`

---

## Issue 9
**Title:** `[P1][BACKEND] Implement FastAPI OCR service`  
**Labels:** `p1`, `backend`, `inference`  
**Assignee:** `@backend-engineer`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

**Acceptance criteria**
- [ ] `/health`, `/detect`, `/recognize` endpoints available.
- [ ] JSON output includes text, bbox, confidence, model_version.
- [ ] Input validation and error handling standardized.

**Tasks**
- [ ] Add `serve.py` service entrypoint.
- [ ] Define Pydantic request/response schemas.
- [ ] Add API integration tests.

---

## Issue 10
**Title:** `[P1][MLE] Add ONNX export + runtime benchmark`  
**Labels:** `p1`, `mle`, `inference`, `reliability`  
**Assignee:** `@mle-engineer`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

**Acceptance criteria**
- [ ] ONNX export path available for detector and recognizer.
- [ ] Latency/throughput report includes p50/p95/p99.
- [ ] Config supports selecting native vs ONNX runtime.

**Tasks**
- [ ] Add export script(s) and runtime switch.
- [ ] Add benchmark harness and report template.
- [ ] Add timeout and fallback policy docs.

---

## EPIC: P1 Human-in-the-Loop Data Flywheel
**Title:** `[EPIC][P1] Add correction workflow and retraining flywheel`  
**Labels:** `epic`, `p1`, `data`, `backend`, `ml`  
**Assignee:** `@product-ml-owner`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

### Checklist
- [ ] User correction payload and review flow implemented.
- [ ] Approved corrections export to training format.
- [ ] Weekly challenger retraining job and report available.

### Child issues
11. `[P1][BACKEND] Implement correction capture + review queue`
12. `[P1][MLE] Build weekly challenger retraining job`

---

## Issue 11
**Title:** `[P1][BACKEND] Implement correction capture + review queue`  
**Labels:** `p1`, `backend`, `data`, `qa`  
**Assignee:** `@backend-engineer`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

**Acceptance criteria**
- [ ] Corrections can be submitted and reviewed.
- [ ] Status transitions tracked (`new`, `approved`, `rejected`).
- [ ] Export includes source metadata + model version.

**Tasks**
- [ ] Define correction schema and storage model.
- [ ] Implement review endpoint/UI placeholder.
- [ ] Add export script for approved records.

---

## Issue 12
**Title:** `[P1][MLE] Build weekly challenger retraining job`  
**Labels:** `p1`, `mle`, `training`, `evaluation`  
**Assignee:** `@mle-engineer`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

**Acceptance criteria**
- [ ] Job produces challenger model weekly.
- [ ] Champion vs challenger report generated automatically.
- [ ] Promotion criteria encoded and visible.

**Tasks**
- [ ] Build scheduled pipeline config.
- [ ] Add metric comparison and alerting.
- [ ] Add promotion/rollback checklist.

---

## EPIC: P2 Security, Governance, and SRE Hardening
**Title:** `[EPIC][P2] Security, governance, and reliability hardening`  
**Labels:** `epic`, `p2`, `security`, `reliability`, `backend`  
**Assignee:** `@platform-owner`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

### Checklist
- [ ] Encryption and retention controls in place.
- [ ] Optional PII redaction support before long-term storage.
- [ ] Canary + rollback runbook tested.

### Child issues
13. `[P2][SECURITY] Add encryption + retention policy enforcement`
14. `[P2][SRE] Canary rollout and rollback runbook`

---

## Issue 13
**Title:** `[P2][SECURITY] Add encryption + retention policy enforcement`  
**Labels:** `p2`, `security`, `backend`  
**Assignee:** `@security-engineer`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

**Acceptance criteria**
- [ ] Data at rest encrypted.
- [ ] Retention window and deletion job enforced.
- [ ] Audit logs available for data deletion events.

**Tasks**
- [ ] Implement storage encryption config.
- [ ] Add retention scheduler.
- [ ] Add deletion audit trail export.

---

## Issue 14
**Title:** `[P2][SRE] Canary rollout and rollback runbook`  
**Labels:** `p2`, `reliability`, `mle`, `backend`  
**Assignee:** `@sre-engineer`  
**Milestone:** `M3: Serving + Flywheel (Weeks 11-16)`

**Acceptance criteria**
- [ ] Canary metrics dashboard defined.
- [ ] Rollback executable in <15 minutes.
- [ ] Incident response playbook documented.

**Tasks**
- [ ] Define SLOs and alert thresholds.
- [ ] Add canary deployment checklist.
- [ ] Add rollback + comms runbook.

---

## Optional import workflow
1. Copy each issue block into GitHub issue creation.
2. Create EPIC issues first, then child issues.
3. Link child issues to EPIC via task list and references.
4. Assign milestones and owners.
5. Add weekly triage cadence (status: `todo`, `in-progress`, `blocked`, `done`).
