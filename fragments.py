# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 12:16:11 2015

@author: itithilien
"""

#### Read/write numbers from/to text  =========================================
with open('p182.txt', 'w') as f:
    for item in setra:
        f.write("%d\n" % item)

# read them back
with open('p229.txt') as f:
    setra = []
    for line in f:
        setra.append(int(line))
#==============================================================================


#### Read/write dict with numberic key/value from/to text =====================
with open('p457a.csv', "w") as f:
    for i in tomato.keys():
        f.write(repr(i) + "," + repr(tomato[i]) + "\n")
#==============================================================================


#### Read/write pickle ========================================================

import pickle
with open('p457diff.pkl', 'wb') as handle:
    pickle.dump(komato, handle)

with open('p457.pkl', 'rb') as handle:
    tomato = pickle.loads(handle.read())

domelo == pomelo
#==============================================================================


#### Read/write csv ===========================================================
import csv
with open('p457.csv', 'wb') as jam:
    kick = csv.writer(jam, delimiter=",", lineterminator="\n")
    for key, val in begio.items():
        kick.writerow([key, val])

pomelo = {}
for key, val in csv.reader(open('p457.csv')):
    pomelo[int(key)] = int(val)

pomelo == lfib

# for writing to file at each loop count, instead of caching the results
with open('p134.csv', 'a') as z:
    writer = csv.writer(z, delimiter=",", lineterminator="\n")
    for i in range(30000, 40000):
        term = str_pfunc(plist[i],plist[i+1])
        row = [i]+[term]
        writer.writerow(row)
        z.flush()

#==============================================================================



#### Read/write json ==========================================================
import json
with open('p078.json', 'w') as f:
    f.write(json.dumps(zebra))

with open('p078.json', 'rb') as handle:
    bjson = json.loads(handle)

bison == zebra
#==============================================================================



#### combination or choose(k, n) ==============================================

x = 1000000
y = 234050
scipy.misc.comb(x, y, exact=True)
gmpy.comb(x, y)
int(sympy.binomial(x, y))
#==============================================================================


#### Rolling sum of list ======================================================
def rollSum(arrai, w):
    "arrai is the list to run a rolling window of w width, summing the elements within"
    return [sum(arrai[i-(w-1):i+1]) if i>(w-1) else sum(arrai[:i+1])  for i in range(len(arrai))]
#==============================================================================


#### Write dictionary to file =================================================
def writeDict(dict, filename, sep):
    with open(filename, "a") as f:
        for i in dict.keys():            
            f.write(i + " " + sep.join([str(x) for x in dict[i]]) + "\n")

writeDict(lfib,'p149.csv',',')
#==============================================================================




#==== Dict of tuple to csv ====================================================
import csv
with open("p457diff.csv", "wb") as f:
    csv.writer(f).writerows((k,) + v for k, v in komato.iteritems())
#==============================================================================



#### Vectorized Totient =======================================================
from loclib import Totient
from itertools import imap

LIMIT = 10**7
totlist = Totient(LIMIT+1)
sunny = list(imap(totlist, range(1,LIMIT+1)))
print sum(imap(totlist, range(1,LIMIT+1)))
#==============================================================================


#### Merge dictionaries =======================================================
x = {'a': 1, 'b': 2}
y = {'b': 3, 'c': 4}
z = x.copy()
z.update(y)
#==============================================================================


