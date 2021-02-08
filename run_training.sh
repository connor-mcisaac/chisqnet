python ./scripts/run_training.py \
    --ifos H1 L1 V1 \
    --frames \
        H1_HOFT_CLEAN_SUB60HZ_C01 \
        L1_HOFT_CLEAN_SUB60HZ_C01 \
        V1Online \
    --channels \
        H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01 \
        L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01 \
        V1:Hrec_hoft_16384Hz \
    --trigger-files \
        /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/full_data/H1-HDF_TRIGGER_MERGE_FULL_DATA-1246824215-3850078.hdf \
        /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/full_data/L1-HDF_TRIGGER_MERGE_FULL_DATA-1246824215-3850078.hdf \
        /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/full_data/V1-HDF_TRIGGER_MERGE_FULL_DATA-1246824215-3850078.hdf \
    --injection-dirs \
        /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/BBHSEOBNRV4HM_INJ_coinc \
        /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/BBHSEOBNRV4_INJ_coinc \
        /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/BBHNRSUR7DQ4_INJ_coinc \
    --injection-approximants \
        SEOBNRv4HM_ROM \
        SEOBNRv4_ROM \
    --segment-files /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/results/1._analysis_time/1.01_segment_data/H1L1V1-INSP_SEGMENTS-1246824215-3850078.xml \
    --foreground-vetos /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/segments/H1L1V1-FOREGROUND_CENSOR-1246824215-3850078.xml \
    --bank /home/koustav.chandra/O3/productions/imbh/a4_initial/a4_INITIAL/bank/H1L1V1-BANK2HDF-1246824215-3850078.hdf \
    --output-file ./data/net_chisq.npy \
    --sample-rate 2048 \
    --data-width 512. \
    --cut-width 16. \
    --snr-width 0.1 \
    --batch-size 16 \
    --batch-num 10 \
    --shift-num 8
