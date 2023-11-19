

CUDA_VISIBLE_DEVICES=4 python3 /mnt/data10t/bakuphome20210617/zhangkun/x_dim_bert/train.py \
  --data_path /mnt/data10t/bakuphome20210617/lz/data/data1/I-T/Flickr30K/ \
  --data_name f30k_precomp \
  --logger_name /mnt/data2/zk/Dim_mask_bert_3/log \
  --model_name /mnt/data2/zk/Dim_mask_bert_3/checkpoint \
  --batch_size 128 \
  --num_epochs=25 \
  --lr_update=15 \
  --learning_rate=.0005 \
  --precomp_enc_type basic \
  --workers 10 \
  --log_step 200 \
  --embed_size 1024 \
  --vse_mean_warmup_epochs 1
