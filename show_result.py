#!/usr/bin/python3
import sys
import helper.ploting_helper as plot

__author__ = 'garrett'

dataset = sys.argv[1]

plot.draw_losses('{}.txt'.format(dataset), '{}'.format(dataset))
