@echo off
setlocal EnableDelayedExpansion

:: conda activate iris_keras

FOR /L %%G IN (1,1,30) DO (
    SET "nPart=%%G"
    FOR /L %%H IN (1,1,4) DO (
    	SET "bins=%%H"
        FOR /L %%I IN (2,1,3) DO (
	    SET "dataset=%%I"
            python .\vgg_full.py -pf "experiments\ndiris_full_vgg_pairs_removebad_!bins!\initial_test\params.json" -d !dataset!  -p !nPart! --use_ndiris --use_pairs -rm !bins!
        )
        FOR /L %%I IN (0,1,3) DO (
	    SET "dataset=%%I"
            python .\vgg_full.py -pf "experiments\gfi_full_vgg_pairs_removebad_!bins!\initial_test\params.json" -d !dataset! -p !nPart! --use_pairs -rm !bins!
        )
    )
)
:: FOR /L nPart IN (1 1); do  # CHANGE NUMBERS APPROPRIATELY
::     FOR /L d IN (2 3); do
::         python .\vgg_full.py -pf experiments\ndiris_full_vgg_pairs\initial_test_sbs\params.json -d "$d" -p "$nPart" -sbs --use_ndiris --use_pairs
::     done
:: done

