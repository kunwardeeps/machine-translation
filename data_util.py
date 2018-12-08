#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 19:26:09 2017

@author: Prince
"""

import json
import unicodedata
import string

with open('jsonData.json') as data_file:
    data = json.load(data_file)
   

de = data.get('de')
for i in range(len(de)):
    de[i] = unicodedata.normalize('NFKD', de[i]).encode('ascii','ignore').translate(None, string.punctuation).lower()


en = data.get('en')
for i in range(len(en)):
    en[i] = unicodedata.normalize('NFKD', en[i]).encode('ascii','ignore').translate(None, string.punctuation).lower()
    
count = 0
en2 = []
de2 = []
for i,j in zip(en,de):
    if(len(i.split()) == len(j.split())):
        if(len(i.split()) < 5):
            p = 5 - len(i.split())
            i = i + p*" enxp"
            j = j + p*" dexp"
            en2.append(i)
            de2.append(j)
            count = count + 1
    
thefile = open('de-json.txt', 'w')
for item in de2:
  thefile.write("%s\n" % item)

     
thefile = open('en-json.txt', 'w')
for item in en2:
  thefile.write("%s\n" % item)


