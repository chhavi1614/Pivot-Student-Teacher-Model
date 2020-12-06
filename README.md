**Data can be downloaded from**
http://opus.nlpl.eu/


**Step 1. preprocess the raw data **

``` python
mkdir data
cd data

mkdir teacher
mkdir student

python ../preprocess.py --train_src path_to_pivot --train_tgt path_to_target \
                  --save_data data/teacher
python ../preprocess.py --train_src path_to_source --train_tgt path_to_pivot \
                  --save_data data/student
```
**Step 2. train teacher model on pivot-to-target **

``` javascript
# the GNMT style teacher model
python train.py -data data/teacher -save_model ./model \
        -layers 8 -rnn_size 1024 -rnn_type GRU \
        -encoder_type brnn -decoder_type rnn \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 128 -batch_type sents -normalization sents  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 1e-3 \
        -max_grad_norm 5 -param_init 0  -param_init_glorot \
        -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 1 -gpu_ranks 0
```
**Step 3. train student model on source-to-pivot **

``` javascript
# the transformer student model
python train.py -data data/student/ -save_model student \
-layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
-encoder_type transformer -decoder_type transformer -position_encoding \
-train_steps 100000 -max_generator_batches 2 -dropout 0.1 \
-batch_size 4096 -batch_type tokens -normalization tokens -accum_count 2 \
-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
-label_smoothing 0.1 -save_checkpoint_steps 10000 \
-world_size 1 -gpu_ranks 0 --teacher_model_path model_step_200000.pt
```

