python train.py \
  --data-path data/etlcdb \
  --etl-dirs ETL1,ETL2,ETL3,ETL4,ETL5,ETL6,ETL7,ETL8G,ETL9G \
  --img-size 64 \
  --batch-size 128 \
  --val-split 0.1 \
  --epochs 20 \
  --lr 1e-3 \
  --pretrained \
  --save-path best_model.pth

