#!/usr/bin/env bash

git diff-files --diff-filter M -z | while read -rd '' meta
do
    read -rd '' file
    mode="${meta%% *}"
    mode="${mode#:1}"
    if [[ "$mode" =~ ^0[0-7]{4}$ ]]
    then
        chmod "$mode" "$file"
    fi
done
