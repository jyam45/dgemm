#!/bin/bash

maxsize=16
tester=./copy2d_unit_test

i=1
while [ $i -le $maxsize ]; do
	j=1
	while [ $j -le $maxsize ]; do
		echo "$tester -m $i -n $j"
		$tester -m $i -n $j
		echo "$tester -m $i -n $j -u"
		$tester -m $i -n $j -u
		echo "$tester -m $i -n $j -t"
		$tester -m $i -n $j -t
		echo "$tester -m $i -n $j -t -u"
		$tester -m $i -n $j -t -u
		j=`expr $j + 1`
	done
	i=`expr $i + 1`
done
