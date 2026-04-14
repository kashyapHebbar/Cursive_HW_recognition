# Prioritized Implementation Backlog

This backlog converts the architecture plan into executable work items.

## Planning assumptions
- Time horizon: 16 weeks.
- Team: 2 ML engineers, 1 MLE/backend engineer, 1 part-time annotator/QA.
- Story points: 1 (tiny) to 13 (very large).
- Priority: P0 (must), P1 (high), P2 (medium).

---

## Epic 1 (P0): Data Foundation & Evaluation Baseline

## Goal
Establish reliable data contracts, writer-aware splits, and OCR metrics (CER/WER) so all future model work is measurable and reproducible.

## Story 1.1: Dataset manifest + validation
**Estimate:** 8 pts  
**Owner:** MLE  
**Dependencies:** none

### Tasks
1. Define dataset manifest schema (`sample_id`, `writer_id`, `split`, `line_transcript`, `image_uri`, quality tags).
2. Add `validate_manifest.py` script for schema checks, missing files, label sanity.
3. Add CI check that fails on manifest violations.

### Acceptance criteria
- Manifest validation catches malformed records and missing image paths.
- Validation report prints counts by split/writer.
- CI blocks merges when validation fails.

## Story 1.2: Writer-aware split generation
**Estimate:** 5 pts  
**Owner:** ML Engineer  
**Dependencies:** Story 1.1

### Tasks
1. Implement split generator by `writer_id` with stratification by text length bucket.
2. Output deterministic split files with a fixed seed.
3. Document split policy.

### Acceptance criteria
- No writer appears in more than one split.
- Split script is deterministic with same seed.
- Split stats exported (samples/writers per split).

## Story 1.3: CER/WER evaluator and baseline report
**Estimate:** 5 pts  
**Owner:** ML Engineer  
**Dependencies:** Story 1.2

### Tasks
1. Implement CER/WER utility script.
2. Add evaluation output (overall + slice metrics).
3. Produce initial benchmark markdown report.

### Acceptance criteria
- Evaluator returns CER/WER for provided predictions and references.
- Slice metrics available by text length and quality tag.
- Baseline report committed.

---

## Epic 2 (P0): Line Detection Pipeline

## Goal
Move from implicit character segmentation to explicit line detection suitable for full-page/line OCR.

## Story 2.1: Detection model integration (DBNet/CRAFT)
**Estimate:** 13 pts  
**Owner:** ML Engineer  
**Dependencies:** Epic 1

### Tasks
1. Add detector module (`detect.py`) with pluggable backend.
2. Implement image-to-line polygon extraction.
3. Save crops + metadata for recognizer input.

### Acceptance criteria
- Detector outputs line boxes/polygons for sample documents.
- Cropped lines are persisted with traceable IDs.
- Detection confidence returned per line.

## Story 2.2: Detection quality harness
**Estimate:** 5 pts  
**Owner:** QA + ML  
**Dependencies:** Story 2.1

### Tasks
1. Add manually labeled subset for detection QA.
2. Compute precision/recall/IoU metrics.
3. Add threshold tuning config.

### Acceptance criteria
- Detection metrics produced on labeled subset.
- Tuning config can shift recall/precision tradeoff.
- Results documented in benchmark file.

---

## Epic 3 (P0): Sequence Recognizer Upgrade

## Goal
Introduce line-level recognition (CRNN+CTC baseline) replacing character-only inference.

## Story 3.1: CRNN+CTC baseline
**Estimate:** 13 pts  
**Owner:** ML Engineer  
**Dependencies:** Epic 1, Epic 2

### Tasks
1. Implement recognizer module (`recognize.py`) with CRNN+CTC.
2. Build training loop with configurable augmentations.
3. Add greedy and beam decoding modes.

### Acceptance criteria
- Model trains end-to-end on line crops.
- Produces line transcripts of variable length.
- CER/WER reported automatically after training.

## Story 3.2: Language-model rescoring
**Estimate:** 8 pts  
**Owner:** ML Engineer  
**Dependencies:** Story 3.1

### Tasks
1. Train/import char/subword LM from domain text.
2. Implement score fusion with tunable alpha/beta.
3. Add ablation report (with vs without LM).

### Acceptance criteria
- LM rescoring available in inference pipeline.
- Ablation shows measurable CER/WER impact.
- Configurable decode parameters documented.

---

## Epic 4 (P1): Inference Service & APIs

## Goal
Expose OCR via a production-friendly API with observability and robust failure handling.

## Story 4.1: FastAPI inference service
**Estimate:** 8 pts  
**Owner:** MLE/Backend  
**Dependencies:** Epic 2, Epic 3

### Tasks
1. Add `serve.py` with endpoints `/health`, `/detect`, `/recognize`.
2. Add request/response schemas with confidence and bbox outputs.
3. Add async batching queue for throughput.

### Acceptance criteria
- API returns structured JSON with line text, bbox, confidence.
- Health endpoint includes model versions.
- Service handles concurrent requests without crashes.

## Story 4.2: Runtime optimization
**Estimate:** 5 pts  
**Owner:** MLE  
**Dependencies:** Story 4.1

### Tasks
1. Export models to ONNX.
2. Benchmark latency and throughput on representative workload.
3. Add dynamic batching and timeout policies.

### Acceptance criteria
- p95 latency and throughput metrics documented.
- ONNX inference path selectable by config.
- Timeout/fallback behavior covered by tests.

---

## Epic 5 (P1): Human-in-the-Loop & Data Flywheel

## Goal
Capture user corrections to continuously improve model quality.

## Story 5.1: Correction capture workflow
**Estimate:** 8 pts  
**Owner:** Backend + QA  
**Dependencies:** Epic 4

### Tasks
1. Add correction payload schema (original text, corrected text, metadata).
2. Store review queue with status transitions.
3. Build export tool for approved corrections.

### Acceptance criteria
- Corrections can be submitted and reviewed.
- Approved corrections exported to training-ready format.
- Audit log includes request/model version.

## Story 5.2: Weekly retraining candidate pipeline
**Estimate:** 5 pts  
**Owner:** MLE  
**Dependencies:** Story 5.1

### Tasks
1. Build job that samples corrected + hard examples.
2. Train challenger model automatically.
3. Compare against champion and produce decision report.

### Acceptance criteria
- Weekly job produces challenger metrics.
- Promotion criteria are explicit and automated.
- Report highlights regressions by slice.

---

## Epic 6 (P2): Governance, Security, and Reliability

## Goal
Harden the system for enterprise usage and safer production operations.

## Story 6.1: Security controls
**Estimate:** 5 pts  
**Owner:** Backend/MLE  
**Dependencies:** Epic 4

### Tasks
1. Encrypt artifacts at rest.
2. Add retention policy + deletion job.
3. Add optional PII redaction pre-storage.

### Acceptance criteria
- Encryption and retention policies enforced in config.
- Deletion logs available for audit.
- PII redaction toggle documented.

## Story 6.2: Reliability & rollback playbook
**Estimate:** 3 pts  
**Owner:** MLE  
**Dependencies:** Epic 4

### Tasks
1. Define SLOs (latency, error rate, confidence drift).
2. Create canary + rollback checklist.
3. Add runbook for incident response.

### Acceptance criteria
- Canary metrics tracked and visible.
- Rollback can be executed in < 15 minutes.
- Incident runbook accessible in repo docs.

---

## Prioritized milestone view

## Milestone A (Weeks 1-4)
- Epic 1 complete
- Story 2.1 started

## Milestone B (Weeks 5-10)
- Epic 2 complete
- Story 3.1 complete
- Story 3.2 started

## Milestone C (Weeks 11-16)
- Epic 3 complete
- Epic 4 complete
- Epic 5 started

## Nice-to-have after week 16
- Epic 5 complete
- Epic 6 complete

---

## Risk register (top 5)

1. **Data quality drift**  
   Mitigation: weekly quality dashboards + correction sampling quotas.
2. **Detection false negatives on noisy scans**  
   Mitigation: threshold tuning and fallback high-recall mode.
3. **LM over-correction of domain-specific words**  
   Mitigation: lexicon constraints + confidence-gated rescoring.
4. **Latency regressions after model upgrades**  
   Mitigation: canary release + ONNX/TensorRT fallback.
5. **Writer-generalization gaps**  
   Mitigation: writer-aware splits + targeted hard-example mining.

---

## Definition of Done (program level)

The upgraded system is considered production-ready when:
1. CER and WER improve by agreed target on holdout data.
2. p95 latency meets SLO under expected load.
3. Confidence calibration supports safe low-confidence routing.
4. Continuous retraining loop operates with auditability.
5. Rollback and incident procedures are tested.


---

For copy/paste issue templates with labels and milestones, use `GITHUB_ISSUES_BACKLOG.md`.
