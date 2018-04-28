#!/bin/bash 

screen -AdmS ex1_kgof -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex1_kmod -X screen -t tabname bash -lic "python ex1_vary_n.py stdnormal_shift_d1"

