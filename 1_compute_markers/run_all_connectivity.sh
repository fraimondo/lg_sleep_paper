#!/bin/zsh
#
b=()
for s in /media/data/lg_meg_sleep/subjects/*;
do
		b+=(${s:t})
done

parallel -S : -j4 --joblog job-connectivity.log --resume --resume-failed --tag --delay 0 'python compute_markers_connectivity.py --runid=20200226_connectivity  --path=/media/data/lg_meg_sleep/ --subject={}' ::: $b
