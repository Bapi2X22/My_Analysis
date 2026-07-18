#!/bin/sh
export X509_USER_PROXY=/afs/cern.ch/user/b/bbapi/.x509up_u177868
. /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh
export PYTHONPATH="/afs/cern.ch/user/b/bbapi/.local/lib/python3.13/site-packages/:$PYTHONPATH"
export PATH="/afs/cern.ch/user/b/bbapi/.local/bin:$PATH"
if [ $1 -eq 0 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-0.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 2 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-2.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 3 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-3.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 4 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-4.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 5 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-5.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 6 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-6.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 7 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-7.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 8 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-8.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 9 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-9.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 10 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-10.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 11 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-11.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 12 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-12.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 13 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-13.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 14 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-14.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 15 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-15.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 16 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-16.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 17 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-17.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 18 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-18.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 19 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-19.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 20 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-20.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 21 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-21.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 22 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-22.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 23 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-23.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 24 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-24.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 25 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-25.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 26 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-26.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 27 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-27.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 28 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-28.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 29 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-29.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 30 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-30.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 31 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-31.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 32 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-32.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 33 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-33.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 34 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-34.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 35 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-35.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 36 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-36.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 37 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-37.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 38 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-38.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 39 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-39.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 40 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-40.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 41 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-41.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 42 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-42.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 43 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-43.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 44 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-44.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 45 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-45.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 46 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-46.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 47 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-47.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 48 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-48.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 49 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-49.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 50 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-50.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 51 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-51.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 52 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-52.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 53 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-53.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 54 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-54.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 55 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-55.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 56 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-56.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 57 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-57.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 58 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-58.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 59 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-59.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 60 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-60.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 61 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-61.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 62 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-62.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 63 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-63.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 64 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-64.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 65 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-65.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 66 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-66.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 67 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-67.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 68 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-68.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 69 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-69.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 70 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-70.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 71 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-71.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 72 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-72.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 73 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-73.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 74 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-74.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 75 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-75.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 76 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-76.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 77 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-77.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 78 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-78.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 79 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-79.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 80 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-80.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 81 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-81.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 82 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-82.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 83 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-83.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 84 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-84.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 85 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-85.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 86 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-86.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 87 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-87.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 88 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-88.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 89 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-89.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 90 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-90.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 91 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-91.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 92 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-92.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 93 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-93.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 94 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-94.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 95 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-95.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 96 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-96.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 97 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-97.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 98 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-98.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 99 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-99.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 100 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-100.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 101 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-101.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 102 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-102.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 103 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-103.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 104 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-104.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 105 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-105.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 106 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-106.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 107 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-107.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 108 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-108.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 109 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-109.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 110 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-110.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 111 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-111.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 112 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-112.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 113 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-113.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 114 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-114.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 115 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-115.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 116 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-116.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 117 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-117.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 118 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-118.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 119 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-119.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 120 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-120.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 121 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-121.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 122 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-122.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 123 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-123.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 124 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-124.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 125 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-125.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 126 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-126.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 127 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-127.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 128 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-128.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 129 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-129.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 130 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-130.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 131 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-131.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 132 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-132.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 133 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-133.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 134 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-134.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 135 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-135.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 136 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-136.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 137 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-137.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 138 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-138.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 139 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-139.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 140 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-140.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 141 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-141.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 142 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-142.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 143 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-143.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 144 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-144.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 145 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-145.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 146 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-146.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 147 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-147.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 148 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-148.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 149 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-149.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 150 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-150.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 151 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-151.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 152 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-152.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 153 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-153.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 154 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-154.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 155 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-155.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 156 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-156.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 157 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-157.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 158 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-158.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 159 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-159.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 160 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-160.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 161 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-161.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 162 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-162.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 163 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-163.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 164 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-164.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 165 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-165.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 166 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-166.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 167 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-167.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 168 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-168.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 169 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-169.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 170 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-170.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 171 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-171.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 172 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-172.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 173 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-173.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 174 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-174.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 175 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-175.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 176 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-176.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 177 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-177.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 178 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-178.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 179 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-179.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 180 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-180.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 181 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-181.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 182 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-182.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 183 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-183.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 184 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-184.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 185 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-185.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 186 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-186.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 187 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-187.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 188 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-188.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 189 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-189.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 190 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-190.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 191 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-191.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 192 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-192.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 193 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-193.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 194 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-194.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 195 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-195.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 196 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-196.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 197 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-197.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 198 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-198.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 199 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-199.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 200 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-200.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 201 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-201.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 202 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-202.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 203 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-203.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 204 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-204.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 205 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-205.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 206 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-206.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 207 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-207.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 208 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-208.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 209 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-209.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 210 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-210.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 211 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-211.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 212 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-212.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 213 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-213.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 214 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-214.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 215 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-215.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 216 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-216.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 217 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-217.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 218 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-218.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 219 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-219.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 220 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-220.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 221 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-221.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 222 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-222.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 223 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-223.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 224 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-224.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 225 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-225.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 226 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-226.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 227 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-227.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 228 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-228.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 229 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-229.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 230 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-230.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 231 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-231.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 232 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-232.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 233 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-233.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 234 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-234.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 235 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-235.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 236 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-236.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 237 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-237.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 238 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-238.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 239 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-239.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 240 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-240.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 241 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-241.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 242 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-242.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 243 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-243.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 244 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-244.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 245 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-245.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 246 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-246.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 247 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-247.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 248 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-248.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 249 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-249.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 250 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-250.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 251 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-251.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 252 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-252.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 253 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-253.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 254 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-254.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 255 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-255.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 256 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-256.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 257 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-257.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 258 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-258.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 259 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-259.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 260 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-260.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 261 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-261.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 262 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-262.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 263 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-263.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 264 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-264.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 265 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-265.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 266 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-266.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 267 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-267.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 268 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-268.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 269 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-269.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 270 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-270.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 271 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-271.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 272 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-272.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 273 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-273.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 274 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-274.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 275 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-275.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 276 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-276.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 277 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-277.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 278 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-278.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 279 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-279.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 280 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-280.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 281 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-281.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 282 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-282.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 283 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-283.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 284 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-284.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 285 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-285.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 286 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-286.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 287 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-287.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 288 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-288.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 289 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-289.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 290 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-290.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 291 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-291.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 292 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-292.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 293 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-293.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 294 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-294.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 295 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-295.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 296 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-296.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 297 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-297.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 298 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-298.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 299 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-299.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 300 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-300.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 301 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-301.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 302 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-302.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 303 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-303.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 304 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-304.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 305 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-305.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 306 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-306.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 307 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-307.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 308 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-308.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 309 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-309.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 310 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-310.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 311 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-311.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 312 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-312.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 313 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-313.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 314 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-314.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 315 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-315.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 316 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-316.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 317 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-317.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 318 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-318.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 319 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-319.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 320 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-320.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 321 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-321.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 322 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-322.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 323 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-323.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 324 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-324.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 325 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-325.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 326 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-326.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 327 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-327.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 328 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-328.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 329 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-329.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 330 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-330.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 331 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-331.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 332 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-332.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 333 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-333.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 334 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-334.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 335 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-335.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 336 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-336.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 337 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-337.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 338 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-338.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 339 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-339.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 340 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-340.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 341 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-341.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 342 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-342.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 343 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-343.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 344 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-344.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 345 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-345.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 346 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-346.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 347 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-347.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 348 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-348.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 349 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-349.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 350 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-350.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 351 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-351.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 352 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-352.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 353 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-353.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 354 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-354.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 355 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-355.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 356 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-356.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 357 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-357.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 358 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-358.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 359 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-359.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 360 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-360.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 361 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-361.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 362 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-362.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 363 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-363.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 364 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-364.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 365 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-365.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 366 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-366.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 367 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-367.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 368 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-368.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 369 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-369.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 370 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-370.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 371 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-371.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 372 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-372.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 373 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-373.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 374 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-374.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 375 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-375.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 376 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-376.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 377 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-377.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 378 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-378.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 379 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-379.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 380 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-380.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 381 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-381.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 382 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-382.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 383 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-383.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 384 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-384.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 385 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-385.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 386 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-386.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 387 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-387.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 388 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-388.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 389 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-389.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 390 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-390.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 391 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-391.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 392 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-392.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 393 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-393.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 394 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-394.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 395 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-395.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 396 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-396.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 397 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-397.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 398 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-398.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 399 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-399.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 400 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-400.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 401 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-401.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 402 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-402.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 403 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-403.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 404 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-404.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 405 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-405.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 406 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-406.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 407 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-407.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 408 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-408.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 409 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-409.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 410 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-410.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 411 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-411.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 412 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-412.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 413 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-413.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 414 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-414.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 415 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-415.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 416 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-416.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 417 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-417.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 418 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-418.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 419 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-419.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 420 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-420.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 421 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-421.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 422 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-422.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 423 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-423.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 424 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-424.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 425 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-425.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 426 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-426.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 427 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-427.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 428 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-428.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 429 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-429.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 430 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-430.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 431 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-431.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 432 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-432.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 433 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-433.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 434 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-434.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 435 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-435.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 436 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-436.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 437 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-437.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 438 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-438.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 439 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-439.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 440 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-440.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 441 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-441.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 442 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-442.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 443 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-443.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 444 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-444.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 445 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-445.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 446 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-446.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 447 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-447.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 448 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-448.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 449 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-449.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 450 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-450.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 451 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-451.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 452 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-452.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 453 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-453.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 454 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-454.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 455 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-455.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 456 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-456.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 457 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-457.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 458 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-458.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 459 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-459.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 460 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-460.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 461 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-461.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 462 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-462.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 463 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-463.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 464 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-464.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 465 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-465.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 466 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-466.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 467 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-467.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 468 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-468.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 469 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-469.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 470 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-470.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 471 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-471.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 472 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-472.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 473 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-473.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 474 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-474.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 475 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-475.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 476 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-476.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 477 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-477.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 478 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-478.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 479 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-479.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 480 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-480.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 481 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-481.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 482 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-482.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 483 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-483.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 484 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-484.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 485 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-485.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 486 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-486.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 487 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-487.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 488 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-488.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 489 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-489.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 490 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-490.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 491 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-491.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 492 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-492.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 493 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-493.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 494 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-494.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 495 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-495.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 496 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-496.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 497 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-497.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 498 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-498.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 499 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-499.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 500 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-500.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 501 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-501.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 502 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-502.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 503 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-503.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 504 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-504.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 505 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-505.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 506 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-506.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 507 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-507.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 508 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-508.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 509 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-509.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 510 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-510.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 511 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-511.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 512 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-512.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 513 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-513.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 514 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-514.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 515 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-515.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 516 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-516.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 517 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-517.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 518 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-518.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 519 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-519.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 520 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-520.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 521 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-521.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 522 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-522.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 523 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-523.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 524 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-524.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 525 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-525.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 526 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-526.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 527 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-527.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 528 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-528.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 529 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-529.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 530 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-530.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 531 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-531.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 532 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-532.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 533 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-533.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 534 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-534.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 535 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-535.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 536 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-536.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 537 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-537.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 538 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-538.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 539 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-539.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 540 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-540.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 541 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-541.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 542 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-542.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 543 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-543.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 544 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-544.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 545 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-545.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 546 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-546.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 547 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-547.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 548 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-548.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 549 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-549.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 550 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-550.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 551 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-551.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 552 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-552.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 553 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-553.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 554 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-554.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 555 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-555.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 556 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-556.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 557 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-557.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 558 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-558.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 559 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-559.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 560 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-560.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 561 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-561.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 562 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-562.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 563 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-563.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 564 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-564.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 565 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-565.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 566 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-566.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 567 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-567.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 568 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-568.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 569 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-569.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 570 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-570.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 571 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-571.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 572 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-572.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 573 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-573.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 574 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-574.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 575 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-575.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 576 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-576.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 577 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-577.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 578 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-578.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 579 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-579.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 580 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-580.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 581 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-581.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 582 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-582.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 583 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-583.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 584 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-584.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 585 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-585.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 586 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-586.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 587 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-587.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 588 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-588.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 589 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-589.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 590 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-590.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 591 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-591.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 592 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-592.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 593 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-593.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 594 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-594.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 595 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-595.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 596 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-596.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 597 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-597.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 598 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-598.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 599 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-599.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 600 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-600.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 601 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-601.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 602 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-602.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 603 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-603.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 604 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-604.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 605 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-605.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 606 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-606.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 607 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-607.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 608 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-608.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 609 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-609.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 610 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-610.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 611 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-611.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 612 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-612.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 613 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-613.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 614 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-614.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 615 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-615.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 616 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-616.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 617 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-617.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 618 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-618.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 619 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-619.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 620 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-620.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 621 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-621.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 622 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-622.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 623 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-623.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 624 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-624.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 625 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-625.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 626 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-626.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 627 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-627.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 628 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-628.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 629 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-629.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 630 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-630.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 631 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-631.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 632 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-632.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 633 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-633.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 634 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-634.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 635 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-635.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 636 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-636.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 637 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-637.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 638 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-638.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 639 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-639.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 640 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-640.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 641 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-641.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 642 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-642.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 643 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-643.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 644 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-644.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 645 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-645.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 646 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-646.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 647 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-647.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 648 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-648.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 649 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-649.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 650 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-650.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 651 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-651.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 652 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-652.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 653 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-653.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 654 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-654.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 655 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-655.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 656 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-656.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 657 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-657.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 658 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-658.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 659 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-659.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 660 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-660.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 661 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-661.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 662 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-662.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 663 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-663.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 664 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-664.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 665 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-665.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 666 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-666.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 667 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-667.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 668 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-668.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 669 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-669.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 670 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-670.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 671 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-671.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 672 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-672.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 673 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-673.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 674 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-674.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 675 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-675.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 676 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-676.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 677 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-677.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 678 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-678.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 679 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-679.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 680 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-680.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 681 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-681.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 682 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-682.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 683 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-683.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 684 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-684.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 685 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-685.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 686 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-686.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 687 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-687.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 688 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-688.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 689 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-689.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 690 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-690.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 691 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-691.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 692 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-692.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 693 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-693.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 694 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-694.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 695 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-695.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 696 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-696.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 697 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-697.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 698 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-698.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 699 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-699.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 700 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-700.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 701 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-701.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 702 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-702.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 703 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-703.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 704 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-704.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 705 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-705.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 706 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-706.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 707 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-707.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 708 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-708.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 709 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-709.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 710 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-710.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 711 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-711.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 712 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-712.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 713 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-713.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 714 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-714.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 715 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-715.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 716 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-716.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 717 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-717.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 718 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-718.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 719 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-719.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 720 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-720.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 721 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-721.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 722 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-722.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 723 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-723.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 724 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-724.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 725 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-725.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 726 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-726.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 727 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-727.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 728 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-728.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 729 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-729.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 730 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-730.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 731 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-731.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 732 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-732.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 733 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-733.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 734 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-734.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 735 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-735.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 736 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-736.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 737 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-737.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 738 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-738.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 739 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-739.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 740 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-740.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 741 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-741.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 742 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-742.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 743 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-743.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 744 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-744.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 745 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-745.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 746 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-746.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 747 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-747.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 748 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-748.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 749 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-749.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 750 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-750.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 751 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-751.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 752 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-752.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 753 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-753.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 754 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-754.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 755 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-755.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 756 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-756.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 757 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-757.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 758 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-758.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 759 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-759.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 760 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-760.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 761 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-761.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 762 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-762.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 763 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-763.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 764 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-764.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 765 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-765.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 766 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-766.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 767 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-767.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 768 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-768.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 769 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-769.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 770 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-770.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 771 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-771.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 772 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-772.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 773 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-773.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 774 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-774.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 775 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-775.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 776 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-776.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 777 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-777.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 778 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-778.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 779 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-779.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 780 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-780.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 781 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-781.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 782 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-782.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 783 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-783.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 784 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-784.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 785 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-785.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 786 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-786.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 787 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-787.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 788 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-788.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 789 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-789.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 790 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-790.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 791 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-791.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 792 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-792.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 793 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-793.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 794 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-794.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 795 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-795.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 796 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-796.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 797 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-797.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 798 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-798.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 799 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-799.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 800 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-800.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 801 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-801.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 802 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-802.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 803 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-803.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 804 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-804.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 805 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-805.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 806 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-806.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 807 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-807.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 808 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-808.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 809 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-809.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 810 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-810.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 811 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-811.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 812 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-812.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 813 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-813.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 814 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-814.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 815 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-815.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 816 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-816.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 817 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-817.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 818 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-818.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 819 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-819.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 820 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-820.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 821 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-821.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 822 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-822.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 823 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-823.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 824 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-824.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 825 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-825.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 826 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-826.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 827 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-827.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 828 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-828.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 829 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-829.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 830 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-830.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 831 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-831.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 832 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-832.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 833 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-833.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 834 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-834.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 835 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-835.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 836 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-836.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 837 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-837.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 838 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-838.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 839 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-839.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 840 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-840.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 841 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-841.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 842 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-842.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 843 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-843.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 844 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-844.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 845 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-845.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 846 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-846.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 847 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-847.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 848 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-848.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 849 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-849.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 850 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-850.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 851 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-851.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 852 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-852.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 853 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-853.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 854 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-854.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 855 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-855.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 856 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-856.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 857 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-857.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 858 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-858.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 859 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-859.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 860 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-860.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 861 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-861.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 862 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-862.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 863 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-863.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 864 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-864.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 865 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-865.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 866 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-866.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 867 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-867.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 868 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-868.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 869 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-869.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 870 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-870.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 871 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-871.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 872 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-872.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 873 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-873.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 874 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-874.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 875 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-875.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 876 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-876.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 877 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-877.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 878 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-878.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 879 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-879.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 880 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-880.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 881 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-881.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 882 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-882.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 883 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-883.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 884 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-884.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 885 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-885.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 886 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-886.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 887 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-887.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 888 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-888.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 889 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-889.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 890 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-890.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 891 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-891.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 892 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-892.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 893 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-893.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 894 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-894.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 895 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-895.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 896 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-896.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 897 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-897.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 898 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-898.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 899 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-899.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 900 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-900.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 901 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-901.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 902 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-902.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 903 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-903.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 904 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-904.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 905 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-905.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 906 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-906.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 907 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-907.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 908 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-908.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 909 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-909.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 910 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-910.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 911 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-911.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 912 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-912.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 913 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-913.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 914 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-914.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 915 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-915.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 916 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-916.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 917 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-917.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 918 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-918.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 919 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-919.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 920 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-920.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 921 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-921.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 922 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-922.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 923 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-923.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 924 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-924.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 925 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-925.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 926 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-926.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 927 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-927.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 928 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-928.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 929 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-929.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 930 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-930.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 931 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-931.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 932 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-932.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 933 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-933.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 934 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-934.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 935 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-935.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 936 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-936.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 937 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-937.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 938 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-938.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 939 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-939.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 940 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-940.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 941 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-941.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 942 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-942.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 943 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-943.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 944 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-944.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 945 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-945.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 946 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-946.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 947 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-947.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 948 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-948.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 949 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-949.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 950 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-950.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 951 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-951.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 952 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-952.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 953 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-953.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 954 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-954.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 955 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-955.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 956 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-956.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 957 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-957.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 958 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-958.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 959 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-959.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 960 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-960.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 961 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-961.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 962 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-962.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 963 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-963.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 964 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-964.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 965 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-965.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 966 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-966.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 967 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-967.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 968 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-968.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 969 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-969.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 970 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-970.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 971 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-971.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 972 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-972.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 973 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-973.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 974 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-974.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 975 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-975.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 976 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-976.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 977 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-977.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 978 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-978.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 979 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-979.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 980 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-980.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 981 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-981.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 982 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-982.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 983 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-983.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 984 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-984.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 985 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-985.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 986 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-986.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 987 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-987.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 988 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-988.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 989 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-989.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 990 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-990.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 991 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-991.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 992 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-992.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 993 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-993.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 994 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-994.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 995 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-995.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 996 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-996.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 997 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-997.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 998 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-998.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 999 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-999.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1000 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1000.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1001 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1001.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1002 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1002.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1003 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1003.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1004 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1004.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1005 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1005.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1006 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1006.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1007 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1007.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1008 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1008.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1009 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1009.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1010 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1010.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1011 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1011.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1012 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1012.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1013 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1013.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1014 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1014.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1015 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1015.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1016 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1016.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1017 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1017.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1018 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1018.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1019 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1019.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1020 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1020.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1021 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1021.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1022 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1022.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1023 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1023.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1024 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1024.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1025 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1025.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1026 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1026.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1027 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1027.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1028 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1028.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1029 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1029.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1030 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1030.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1031 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1031.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1032 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1032.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1033 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1033.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1034 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1034.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1035 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1035.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1036 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1036.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1037 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1037.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1038 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1038.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1039 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1039.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1040 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1040.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1041 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1041.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1042 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1042.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1043 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1043.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1044 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1044.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1045 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1045.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1046 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1046.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1047 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1047.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1048 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1048.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1049 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1049.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1050 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1050.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1051 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1051.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1052 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1052.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1053 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1053.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1054 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1054.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1055 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1055.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1056 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1056.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1057 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1057.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1058 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1058.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1059 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1059.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1060 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1060.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1061 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1061.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1062 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1062.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1063 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1063.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1064 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1064.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1065 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1065.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1066 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1066.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1067 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1067.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1068 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1068.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1069 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1069.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1070 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1070.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1071 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1071.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1072 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1072.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1073 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1073.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1074 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1074.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1075 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1075.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1076 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1076.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1077 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1077.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1078 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1078.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1079 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1079.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1080 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1080.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1081 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1081.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1082 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1082.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1083 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1083.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1084 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1084.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1085 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1085.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1086 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1086.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1087 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1087.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1088 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1088.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1089 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1089.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1090 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1090.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1091 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1091.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1092 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1092.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1093 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1093.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1094 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1094.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1095 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1095.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1096 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1096.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1097 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1097.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1098 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1098.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1099 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1099.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1100 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1100.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1101 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1101.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1102 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1102.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1103 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1103.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1104 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1104.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1105 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1105.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1106 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1106.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1107 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1107.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1108 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1108.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1109 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1109.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1110 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1110.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1111 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1111.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1112 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1112.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1113 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1113.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1114 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1114.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1115 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1115.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1116 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1116.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1117 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1117.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1118 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1118.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1119 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1119.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1120 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1120.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1121 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1121.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1122 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1122.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1123 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1123.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1124 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1124.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1125 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1125.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1126 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1126.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1127 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1127.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1128 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1128.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1129 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1129.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1130 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1130.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
if [ $1 -eq 1131 ]; then
    /usr/bin/env run_analysis.py --json-analysis /eos/home-b/bbapi/My_Analysis/2024_efficiency_study/Data/configs/.higgs_dna_vanilla_lxplus/my_runner_2024_20260715_213410/inputs/AN-Muon-Data-2024F-1131.json -d /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Data/NTuples_2024_HDNA_presel --triggerGroup .*EGamma.* --nano-version 15 --skipbadfiles --executor iterative || exit 107
    exit 0
fi
