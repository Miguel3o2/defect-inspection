Subject: AI/ML Engineer application: end-to-end defect inspection pipeline

Hi [Name],

I'm applying for the AI/ML Engineer role. Your JD stresses choosing the right model instead of defaulting to deep learning, and owning the pipeline end to end, so I built a project around that: a surface-defect inspection pipeline on NEU-DET and MVTec AD.

On NEU-DET, a handcrafted-feature LightGBM baseline reaches {{baseline_test_f1}} test macro-F1 against {{deep_test_f1}} for a fine-tuned ResNet. [One sentence, in your own words, on what that taught you.] PatchCore, trained on defect-free images only, reaches {{patchcore_image_auroc}} mean image AUROC on MVTec AD. The ONNX Runtime service has {{ort_p50_ms}} ms p50 latency versus {{torch_p50_ms}} ms in PyTorch on CPU ({{latency_reduction_pct}}% lower). The repo also has CI, a failure analysis with label-noise cleanup, and drift monitoring that I tested with simulated blur and brightness shifts.

Repo: [link] | Failure analysis: [link]

I have [X years] of experience in [one-line relevant background]. Could we find 15 minutes this week or next?

[Name]
