#!/bin/bash 

screen -AdmS ex1_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex1_kmod -X screen -t tab0 bash -lic "python ex2_prob_params.py stdnorm_shift_d1"


