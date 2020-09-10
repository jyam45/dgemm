#!/bin/bash

maxsize=16
tester=./unit_test

i=1
while [ $i -le $maxsize ]; do
	j=1
	while [ $j -le $maxsize ]; do
		k=1
		while [ $k -le $maxsize ]; do
			echo "$tester -m $i -n $j -k $k"
			$tester -m $i -n $j -k $k
			k=`expr $k + 1` 
		done
		j=`expr $j + 1`
	done
	i=`expr $i + 1`
done
