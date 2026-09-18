#!/bin/bash

python3 event_selector.py --input-dir /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTto2L2Nu_24SummerRun3/ --output /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTto2L2Nu_24SummerRun3_selected --chunk-size 100

python3 event_selector.py --input-dir /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTtoLNu2Q_24SummerRun3/ --output /eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTtoLNu2Q_24SummerRun3_selected --chunk-size 100
