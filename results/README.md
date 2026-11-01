# Results

`pred_test_e2b_test_<FD>_<model>_s<seed>.csv` are the one-shot test predictions
(engine, rul_true, rul_pred). `e2b_test_*.json` are the matching metrics
(cap-125 RMSE and unclipped RMSE). `e2b_FD*.json` are the training/validation/gate
summaries for the five seeds. The ablation, mechanism, robustness, stability, and
trained-norm summaries are included as JSON.

Checkpoints are not stored in the repository (size); they can be regenerated with
`code/train_e2.py` and the seeds in `seeds.yaml`.
