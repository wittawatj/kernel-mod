#!/bin/bash 

screen -AdmS ex1_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex1_kmod -X screen -t tab1 bash -lic "python ex1_vary_n.py stdnormal_h0_d1"
#screen -S ex1_kmod -X screen -t tab1 bash -lic "python ex1_vary_n.py stdnormal_h0_d5"
#screen -S ex1_kmod -X screen -t tab0 bash -lic "python ex1_vary_n.py stdnorm_shift_d1"
#screen -S ex1_kmod -X screen -t tab0 bash -lic "python ex1_vary_n.py stdnorm_shift_d20"
screen -S ex1_kmod -X screen -t tab2 bash -lic "python ex1_vary_n.py stdnorm_h0_d50"


