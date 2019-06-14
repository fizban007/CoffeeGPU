#!/usr/bin/env bash

mkdir -p $2
cd $1 || exit
for fl in *
do
  # cpp -nostdinc -C -P -w $2 $fl > $(addprefix $3 $(notdir $fl))
  sed "s/^[ \t]*#/#/" "$fl" > "../$2/$fl"
done
