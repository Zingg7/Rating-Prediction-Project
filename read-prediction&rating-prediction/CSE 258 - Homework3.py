# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:18:35 2019

@author: Administrator
"""

import gzip
import csv
import numpy as np

# Read the csv
c = csv.reader(gzip.open('./assignment1.tar.gz', 'rt', encoding="utf8"), delimiter = '\t')
dataset = []

header = f.readline()
for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    dataset.append(d)