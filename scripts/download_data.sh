#!/usr/bin/env bash
# Downloads MNIST and Fashion-MNIST (idx format) into data/. UCI Digits ships with scikit-learn.
set -e
cd "$(dirname "$0")/../data"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -fL -o "mn_$f" "https://raw.githubusercontent.com/fgnt/mnist/master/$f"
  curl -fL -o "fm_$f" "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f"
done
echo "Data ready in $(pwd)"
