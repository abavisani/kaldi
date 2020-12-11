#!/usr/bin/env bash

# Siyuan Feng: based on wsj/s5/local/chain/tuning/run_tdnn_1g.sh
# neural network layers are resnet-style TDNN-F model.
set -e -o pipefail

# Note: to run this with models trained on language-suffix-word-level supervisions,
# use data/lang_universal_test instead
#langdir=data/lang_combined_test
lang_to_recog=Czech


data_aug_suffix= # set to null, whereas typically kaldi uses _sp
lang_dir_suffix= # set to null or _1gram

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=14
stop_stage=15
align_fmllr_stage=3
num_epochs=4
get_egs_stage=0
nj=30
iter_to_decode=final
nj_decode=1
nt_decode=1
nj_align_fmllr_lats=100
ivector_extract_nj=1
num_jobs_initial=4
num_jobs_final=4
#train_set=train
gmm=tri5 # the gmm for the target data
num_threads_ubm=12
nnet3_affix= # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
dropout_schedule='0,0@0.20,0.3@0.50,0'
minibatch_size=128
remove_egs=false
# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
train_exit_stage=10000
tree_affix=   # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=1g #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
#common_egs_dir=  # you can set this to use previously dumped egs.
frames_per_eg=150,120,90,75
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
gpu_decode=false

# End configuration section.
echo "$0 $@" # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
langs_config=conf/experiments/crossling_eval_${lang_to_recog}.conf
if [ $langs_config ]; then
  # shellcheck disable=SC1090
  source $langs_config
  echo "Done getting language config from $langs_config"
else
  # BABEL TRAIN:
  # Amharic - 307
  # Bengali - 103
  # Cantonese - 101
  # Javanese - 402
  # Vietnamese - 107
  # Zulu - 206
  # BABEL TEST:
  # Georgian - 404
  # Lao - 203

  #GlobalPhone:
  #Czech       S0196
  #French      S0197
  #Spanish     S0203
  #mandarin    S0193
  #Thai        S0321
  babel_langs="307 103 101 402 107 206 404 203"
  babel_recog="${babel_langs}"
  gp_langs="Czech French Mandarin Spanish Thai"
  gp_recog="${gp_langs}"
  gp_path="/export/corpora5/GlobalPhone"
  mboshi_train=false
  mboshi_recog=false
  gp_romanized=false
fi
dir_suffix=_crosslang_recog_${lang_to_recog}
langdir=data/lang_combined${dir_suffix}
train_set=universal${dir_suffix}/train
#train_set=""
#train_set_hires=""
dev_set=""
#dev_set_hires=""
for l in ${babel_langs}; do
  #  train_set="$l/data/train_${l} ${train_set}"
  #  train_set_hires="$l/data_mfcc_hires/train_${l} ${train_set_hires}"

  dev_set="$l/data/dev_${l} ${dev_set}"
  #  dev_set_hires="$l/data_mfcc_hires/dev_${l} ${dev_set_hires}"
done

for l in ${gp_langs}; do
  #  train_set="GlobalPhone/gp_${l}_train ${train_set}"
  dev_set="GlobalPhone/gp_${l}_dev ${dev_set}"
done

#train_set=${train_set%% }
#train_set_hires=${train_set_hires%% }

dev_set=${dev_set%% }
#dev_set_hires=${dev_set_hires%% }

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

echo "Training data directories: $train_set"
echo "Dev data directories: ${dev_set[*]}"
echo "Eval data directories: ${recog_set[*]}"

full_train_set=train
full_dev_set=dev

function langname() {
  # Utility
  echo "$(basename "$1")"
}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

local/chain_crosslang/run_ivector_common.sh --stage $stage --stop-stage $stop_stage \
  --nj $nj \
  --lang-to-recog $lang_to_recog \
  --gmm $gmm \
  --num-threads-ubm $num_threads_ubm \
  --nnet3-affix "$nnet3_affix"

data_dir=$train_set
lang_name=universal${dir_suffix}

gmm_dir=exp/gmm${dir_suffix}/$gmm
ali_dir=exp/gmm${dir_suffix}/${gmm}_ali
tree_dir=exp/chain${nnet3_affix}/$lang_name/tree${tree_affix}
lat_dir=exp/chain${nnet3_affix}/$lang_name/${gmm}${data_aug_suffix}_lats
dir=exp/chain${nnet3_affix}/$lang_name/tdnn${tdnn_affix}${data_aug_suffix}
train_data_dir=data/${data_dir}${data_aug_suffix}_hires
lores_train_data_dir=data/${data_dir}
train_ivector_dir=exp/nnet3${nnet3_affix}/$lang_name/ivectors${data_aug_suffix}_hires

common_egs_dir= #exp/chain${nnet3_affix}/$lang_name/tdnn1b${data_aug_suffix}/egs  # you can set this to use previously dumped egs.
for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
  $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 14 ] && [ $stop_stage -gt 14 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  #1. data/lang_universal; 2. data/langp/lang_universal; 3. data/lang_chain/lang_universal
  if [ -d data/lang_chain/lang_${lang_name} ]; then
    if [ data/lang_chain/lang_$lang_name/L.fst -nt data/lang_$lang_name/L.fst ]; then
      echo "$0: data/lang_chain/lang_$lang_name already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain/lang_$lang_name already exists and seems to be older than data/lang/$lang_name ..."
      echo " ... not sure what to do.  Exiting."
      exit 1
    fi
  else
    mkdir -p data/lang_chain
    cp -r $langdir data/lang_chain/lang_$lang_name
    silphonelist=$(cat data/lang_chain/lang_$lang_name/phones/silence.csl) || exit 1
    nonsilphonelist=$(cat data/lang_chain/lang_$lang_name/phones/nonsilence.csl) || exit 1
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/lang_$lang_name/topo
  fi
fi

if [ $stage -le 15 ] && [ $stop_stage -gt 15 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --stage $align_fmllr_stage --nj $nj_align_fmllr_lats --cmd "$train_cmd" ${lores_train_data_dir} \
    $langdir $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 16 ] && [ $stop_stage -gt 16 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1
  fi
  # uses $num_jobs from $ali_dir/num_jobs
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --leftmost-questions-truncate -1 \
    --cmd "$train_cmd" 4000 ${lores_train_data_dir} data/lang_chain/lang_$lang_name $ali_dir $tree_dir
fi

xent_regularize=0.1

if [ $stage -le 17 ] && [ $stop_stage -gt 17 ]; then
  echo "$0: creating neural net configs using the xconfig parser"
  feat_dim=$(feat-to-dim scp:${train_data_dir}/feats.scp -)
  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true" #"l2-regularize=0.002"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.0005"

  mkdir -p $dir/configs
  cat <<EOF >$dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=$feat_dim name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024

  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3

  linear-component name=prefinal-l dim=192 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 18 ] && [ $stop_stage -gt 18 ]; then
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
  #  utils/create_split_dir.pl \
  #   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  #fi
  echo "exit stage is $train_exit_stage"
  mkdir -p $dir/egs/ || exit 1;
  touch $dir/egs/.nodelete
  steps/nnet3/chain/train.py --stage $train_stage --exit-stage $train_exit_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir ${train_data_dir} \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir || exit 1
fi

if [ $stage -le 19 ] && [ $stop_stage -gt 19 ]  ; then
  echo "$0: creating high-resolution MFCC features for the test data"
  mfccdir=mfcc${data_aug_suffix}_hires${dir_suffix}

  # Feature extraction
  for data_dir in ${recog_set}; do
    (
      lang_name=$(langname $data_dir)
      utils/copy_data_dir.sh data/$data_dir data/${data_dir}_hires
      steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" \
        data/${data_dir}_hires exp/make_mfcc_hires/${data_dir} $mfccdir
      steps/compute_cmvn_stats.sh data/${data_dir}_hires
      utils/fix_data_dir.sh data/${data_dir}_hires

      utils/data/limit_feature_dim.sh 0:39 \
        data/${data_dir}_hires data/${data_dir}_hires_nopitch || exit 1;
      steps/compute_cmvn_stats.sh \
        data/${data_dir}_hires_nopitch exp/make_mfcc_hires/${data_dir}_nopitch $mfccdir || exit 1 

    ) &
    sleep 2
  done
  wait
fi

if [ $stage -le 20 ]  && [ $stop_stage -gt 20 ] ; then
  echo "$0: creating ivectors for the test data"
  for data_dir in ${recog_set}; do
    #nspk=$(wc -l <data/${data_dir}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $ivector_extract_nj \
      data/${data_dir}_hires_nopitch exp/nnet3/universal${dir_suffix}/extractor \
      exp/nnet3/universal${dir_suffix}/ivectors_${data_dir}_hires
  done
fi

#LMTYPE=phn_bg_cross

if [ "$lang_dir_suffix" = "_1gram" ]; then
  LMTYPE=phn_ug_cross
elif [ "$lang_dir_suffix" = "_3gram" ]; then
  LMTYPE=phn_tg_cross
elif [ "$lang_dir_suffix" = "_4gram" ]; then
  LMTYPE=phn_fg_cross
else
  LMTYPE=phn_bg_cross
fi


if [ $stage -le 21 ] && [ $stop_stage -gt 21 ]  ; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_chain/lang_${lang_name}/phones.txt ${langdir}_test${lang_dir_suffix}/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 ${langdir}_test${lang_dir_suffix} \
    $tree_dir $tree_dir/graph_${LMTYPE} || exit 1
fi

if [ $stage -le 22 ] && [ $stop_stage -gt 22 ]  ; then
  frames_per_chunk=$(echo $frames_per_eg | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $recog_set; do
    (
      data_affix=$(echo $data | sed s/eval_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)


      # (pzelasko): Eventually we'll have more LM types here, for now it's just one
      if [ "$iter_to_decode" = "final" ]; then
         decoder_affix=""
      else
         decoder_affix="_iter${iter_to_decode}"
      fi
      for lmtype in $LMTYPE; do
        steps/nnet3/decode.sh \
          --iter $iter_to_decode \
          --use-gpu $gpu_decode \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 0 --extra-right-context 0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nj_decode  --num-threads $nt_decode \
          --online-ivector-dir exp/nnet3/universal${dir_suffix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix}${decoder_affix} || exit 1
      done
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

echo "$0: succeeded"
