#!/bin/bash
# program number
PROGNUM=1
PROGNUMPAD=`echo $PROGNUM | awk '{printf("%04d\n", $1)}'`

# TARGDIR=/global/homes/m/mfhe/github/desi2-forecast/files/$PROGNUMPAD
TARGDIR=$1

TILEFN=/home/hmf/work/FA_test/data/random/targets_exdense_box5_00_10000.fits
FADIR=$TARGDIR

# fa
SURVEY=main # ! for standards only !
DTVER=1.1.1
HDRSURVEY=special # what will be recorded in the fiberassign header


echo PROGNUM=$PROGNUM
echo TARGDIR=$TARGDIR
echo TILEFN=$TILEFN
echo FADIR=$FADIR

#godesi 22.2
# loaded desitarget version is 2.4.0, so ok
# module swap fiberassign/5.5.0
export DESIMODEL=$HOME/work/FA_test/desi
# export SKYHEALPIXS_DIR=$DESI_ROOT/target/skyhealpixs/v1

# grabbing some fiberassign settings from TARGFN header
TARGFN=`python -c 'from fiberassign.fba_tertiary_io import get_targfn; print(get_targfn('$PROGNUM', targdir="'$TARGDIR'"))'`
echo TARGFN=$TARGFN
TMPFN=`mktemp`
fitsheader -e 1 -k FAPRGRM -k OBSCONDS -k SBPROF -k GOALTIME --table ascii.csv $TARGFN > $TMPFN

HDRFAPRGRM=`grep FAPRGRM $TMPFN | awk -F "," '{print $NF}'`
OBSCONDS=`grep OBSCONDS $TMPFN | awk -F "," '{print $NF}'`
SBPROF=`grep SBPROF $TMPFN | awk -F "," '{print $NF}'`
GOALTIME=`grep GOALTIME $TMPFN | awk -F "," '{print $NF}'`

# HDRFAPRGRM=tertiary1
# OBSCONDS='DARK'
# SBPROF='ELG'
# GOALTIME=1200

echo HDRFAPRGRM=$HDRFAPRGRM
echo OBSCONDS=$OBSCONDS
echo SBPROF=$SBPROF
echo GOALTIME=$GOALTIME

# ! custom tiles*ecsv format !
# get the TILEID, RA, DEC, HA column numbers
LINE=`grep TILEID $TILEFN | grep -v \#`
TILEIDCOL=`echo $LINE | awk '{for (i=1; i<=NF; i++) if ($i == "TILEID") print i;}'`
TILERACOL=`echo $LINE | awk '{for (i=1; i<=NF; i++) if ($i == "RA") print i;}'`
TILEDECCOL=`echo $LINE | awk '{for (i=1; i<=NF; i++) if ($i == "DEC") print i;}'`
TILEHACOL=`echo $LINE | awk '{for (i=1; i<=NF; i++) if ($i == "DESIGNHA") print i;}'`

# tiles
grep -v \# $TILEFN | grep -v TILEID > $TMPFN
#NTILE=`wc -l $TMPFN | awk '{print $1}'`
NTILE=$2

for i in $(seq 1 $NTILE)
do
    # tile properties
    TILEID=`awk '{if (NR == '$i') print $'$TILEIDCOL'}' $TMPFN`
    TILERA=`awk '{if (NR == '$i') print $'$TILERACOL'}' $TMPFN`
    TILEDEC=`awk '{if (NR == '$i') print $'$TILEDECCOL'}' $TMPFN`
    TILEHA=`awk '{if (NR == '$i') print $'$TILEHACOL'}' $TMPFN`

    TILEIDPAD=`echo $TILEID | awk '{printf("%06d\n", $1)}'`

    TOOFN=`python -c 'from fiberassign.fba_tertiary_io import get_toofn; print(get_toofn('$PROGNUM', '$TILEID', targdir="'$TARGDIR'"))'`
    LOGFN=`echo $TOOFN | sed -e 's/.ecsv/.log/'`

    OUTFILE=$FADIR/082/fiberassign-$(printf "%06d" $TILEID).fits.gz
    echo OUTFILE=$OUTFILE

    # —— 如果 OUTFILE 已存在，跳过整个 tile —— #
    if [[ -f "$OUTFILE" ]]; then
        echo "Tile $TILEID already has fiberassign output ($OUTFILE), skipping entire tile..."
        continue
    fi

    # 执行 fba_tertiary_too
    if [[ i -eq 1 ]]; then
        CMD="fba_tertiary_too --tileid $TILEID --tilera $TILERA --tiledec $TILEDEC --targdir $TARGDIR --fadir $FADIR --prognum $PROGNUM > $LOGFN 2>&1"
        echo $CMD
        eval $CMD
    else
        PREVTILEIDS=`awk '{if (NR == 1) print $1}' $TMPFN`
        for j in $(seq 2 `echo $i | awk '{print $1-1}'`)
        do
            PREVTILEID=`awk '{if (NR == '$j') print $1}' $TMPFN`
            PREVTILEIDS="$PREVTILEIDS,$PREVTILEID"
        done
        echo $TILEID $TILERA $TILEDEC $PREVTILEIDS
        CMD="fba_tertiary_too --tileid $TILEID --tilera $TILERA --tiledec $TILEDEC --targdir $TARGDIR --fadir $FADIR --prognum $PROGNUM --previous_tileids $PREVTILEIDS > $LOGFN 2>&1"
        echo $CMD
        eval $CMD
    fi
    python /global/homes/m/mfhe/github/desi2-forecast/notebook/shuffle_too_for_128realizations.py $TOOFN $TOOFN
    # —— 执行 fba_launch —— #
    CMD="fba_launch --outdir $FADIR --tileid $TILEID --tilera $TILERA --tiledec $TILEDEC --ha $TILEHA --survey $SURVEY --program $OBSCONDS --sbprof $SBPROF --goaltime $GOALTIME --dtver $DTVER --hdr_survey $HDRSURVEY --hdr_faprgrm $HDRFAPRGRM --nosteps scnd --targ_std_only --too_tile --forcetileid y --custom_too_file $TOOFN --custom_too_development"
    echo $CMD
    eval $CMD

done