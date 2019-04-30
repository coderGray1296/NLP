#!/bin/bash

while read line; do
    if [[ ${line:0:8} != '## Tweet' ]]; then
        echo $line | awk '{printf "%s\t%s\n", $1, $2}'
    fi
done
