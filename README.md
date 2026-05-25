Dataset and linear SVM demo

Files added:
- `data/svm_hours.csv`: CSV with columns `x,y` (50 rows)
- `data/svm_hours_train.csv`: 80% training split (40 rows)
- `data/svm_hours_test.csv`: 20% testing split (10 rows)
- `train_svm.py`: simple script to train a linear SVM from scratch and print model info
- `requirements.txt`: empty (no external deps)

Quick start (PowerShell):

```powershell
python train_svm.py
```

Notes:
- The dataset uses -1 for Fail and +1 for Pass because SVM formulations work naturally with labels -1 and +1.
- The demo implements a basic linear SVM (hinge loss + L2 regularization) without external libraries.
