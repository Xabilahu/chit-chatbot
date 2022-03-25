#!/bin/bash

inDir=json_data_100
outDir=json_data_100_yml

if [ ! -d ${outDir} ] ; then
    mkdir ${outDir}
fi

if [ ! -d ${inDir} ] ; then
	echo "Input directory ${inDir} does not exist!"
	exit 1
fi

cmd="python3 json2yml.py"

for fname in ${inDir}/*.json; do
	bname=`basename ${fname} .json`
	echo "${cmd} ${fname} > ${outDir}/${bname}.yml"
	${cmd} ${fname} > ${outDir}/${bname}.yml
done
