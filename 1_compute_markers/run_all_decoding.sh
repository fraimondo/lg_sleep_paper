#!/bin/zsh
#
b=()
for s in /media/data/lg_meg_sleep/subjects/*;
do
		b+=(${s:t})
done

parallel -S : -j4 --joblog job-decoding.log --resume --resume-failed --tag --delay 0 'python compute_markers_decoding.py --runid=20200226_decoding  --path=/media/data/lg_meg_sleep/ --subject={}' ::: $b
