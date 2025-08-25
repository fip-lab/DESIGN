#!/bin/bash

nohup python3 score.py --resultSavePath 'Bart-large_withABSA_withSelect_1'  > Bart-large_withABSA_withSelect_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-large_withABSA_withSelect_2'  > Bart-large_withABSA_withSelect_2.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-large_withABSA_1'  > Bart-large_withABSA_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-large_withABSA_2'  > Bart-large_withABSA_2.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-large_withSelect_1'  > Bart-large_withSelect_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-large_withSelect_2'  > Bart-large_withSelect_2.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-large_1'  > Bart-large_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-large_2'  > Bart-larget_2.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_withABSA_withSelect_1'  > Bart-base_withABSA_withSelect_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_withABSA_withSelect_2'  > Bart-base_withABSA_withSelect_2.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_withABSA_1'  > Bart-base_withABSA_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_withABSA_2'  > Bart-base_withABSA_2.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_withSelect_1'  > Bart-base_withSelect_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_withSelect_2'  > Bart-base_withSelect_2.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_1'  > Bart-base_1.log 2>&1 &
nohup python3 score.py --resultSavePath 'Bart-base_2'  > Bart-base_2.log 2>&1 &