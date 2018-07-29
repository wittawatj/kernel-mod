#!/bin/bash 

screen -AdmS ex3_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex3_kmod -X screen -t tab0 bash -lic "python ex3_real_images.py clba_p_gs_q_gs_r_rs"
screen -S ex3_kmod -X screen -t tab0 bash -lic "python ex3_real_images.py clba_p_rs_q_rs_r_rs"
screen -S ex3_kmod -X screen -t tab0 bash -lic "python ex3_real_images.py clba_p_gs_q_gn_r_rs"
screen -S ex3_kmod -X screen -t tab0 bash -lic "python ex3_real_images.py clba_p_gs_q_gn_r_rn"
screen -S ex3_kmod -X screen -t tab0 bash -lic "python ex3_real_images.py clba_p_gs_q_gn_r_rm"
screen -S ex3_kmod -X screen -t tab0 bash -lic "python ex3_real_images.py clba_p_rs_q_rn_r_rm"
#screen -S ex3_kmod -X screen -t tab0 bash -lic "python ex3_real_images.py cf10_p_hd_q_dd_r_ad"


