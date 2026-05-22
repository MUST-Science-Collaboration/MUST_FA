#!/bin/bash

# This documents how the focalplane files were generated.
# it was run from the parent directory of the fp_settings
# svn checkout.  Missing positioner data was filled with
# nominal fake ones.
#
# The petal ID to location mapping comes from DocDB 3596
#

#
pushd $(dirname $0) > /dev/null
thisdir=$(pwd)
popd > /dev/null

desi_generate_focalplane \
    --pos_settings ./fp_settings/pos_settings \
    --petal_id2loc '4:0,5:1,6:2,3:3,8:4,10:5,11:6,2:7,7:8,9:9' \
    --fillfake \
    --startvalid '2019-09-16T00:00:00'

