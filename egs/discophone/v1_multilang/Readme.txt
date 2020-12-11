Ali:
In this version, I use speed perturbation on top of Siyuan's script in local/chain_multilang/tuning/run_tdnn_1g.sh. 
Edited scripts include:
 /local/chain_multilang/run_ivector_common.sh -> uncomment sp perturbatio + use mfcc instead of plp feats
 /utils/data/get_reco2dur.sh  -> line 90: read_entire_file=true

====== Readme before =================
v1_multilang/ is developed based on v1/.
v1_multilang uses all 8 Babel languages and 5 GlobalPhone languages to train hybrid TDNN-HMM systems in a multilingual manner.

Main differences are:
1. ./run.sh modified 
2. ./local/setup_languages_multilang.sh added to add language-specific suffices to every word of each language.
