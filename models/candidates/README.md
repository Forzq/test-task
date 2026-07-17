# Candidate model checkpoints

This directory stores trained candidates that have not yet replaced the deployed models.

## classifier_hardneg_v2.pt

- Source: `runs/classify/crop_verifier_hardneg_v2/weights/best.pt`
- Training data: balanced `bolt`, `nut`, and `other` crop dataset with prioritised hard negatives
- Best reported validation top-1 accuracy: `0.98663`
- Status: candidate; threshold calibration and regression comparison are required before deployment

The deployed classifier remains `models/classifier_best.pt` until this candidate passes validation.
