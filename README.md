
---

# LLP_DC Training Configuration

| Argument         | Type    | Default    | Description                  |
| ---------------- | ------- | ---------- | ---------------------------- |
| `--root`         | `str`   | `./data`   | Dataset directory            |
| `--wresnet-k`    | `int`   | `2`        | Width factor of WideResNet   |
| `--wresnet-n`    | `int`   | `28`       | Depth of WideResNet          |
| `--dataset`      | `str`   | `CIFAR100` | Dataset name                 |
| `--n-classes`    | `int`   | `100`      | Number of classes            |
| `--n-epoches`    | `int`   | `1024`     | Training epochs              |
| `--batchsize`    | `int`   | `64`       | Bag batch size               |
| `--bagsize`      | `int`   | `16`       | Instances per bag            |
| `--eval-ema`     | `bool`  | `False`    | Use EMA model for evaluation |
| `--ema-m`        | `float` | `0.999`    | EMA decay rate               |
| `--lr`           | `float` | `0.03`     | Learning rate                |
| `--weight-decay` | `float` | `1e-3`     | Weight decay                 |
| `--momentum`     | `float` | `0.9`      | Momentum                     |
| `--seed`         | `int`   | `10`       | Random seed                  |
| `--lam-u`        | `float` | `0.5`      | Proportion loss weight       |
| `--thr`          | `float` | `0.6`      | Pseudo label threshold       |
| `--exp-dir`      | `str`   | `LLP_DC`   | Experiment directory         |
| `--checkpoint`   | `str`   | `''`       | Path to pretrained model     |
| `--folds`        | `str`   | `2`        | Dataset folds                |

---

### Example Command

```bash
python LLP_DC.py \
  --dataset CIFAR100 \
  --n-classes 100 \
  --batchsize 64 \
  --bagsize 16 \
  --lr 0.03 \
  --lam-c 1.0 \
  --lam-u 0.5 \
  --thr 0.6 \
  --n-epoches 1024 \
  --exp-dir results_cifar100
```】
