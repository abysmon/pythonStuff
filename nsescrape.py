# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:56:51 2015

@author: itithilien
"""

from nsetools import Nse
import pandas as pd
import time
from urllib2 import build_opener, HTTPCookieProcessor, Request

#test
nse = Nse()
print nse

all_stock_codes = nse.get_stock_codes()
ticklist = all_stock_codes.keys()
ticklist = sorted(ticklist)


while True:
    sata = pd.DataFrame()
    for ticker in ticklist:
        ptime = time.ctime()
        try:
            q = nse.get_quote(ticker)
            qdf = pd.DataFrame(q.items(), columns=['Item', 'Value'])
            qdf['symbol'] = ticker
            qdf['timestamp'] = ptime
            #qdf = qdf.sort(['Item'])
            sata = pd.concat([sata,qdf],axis=0,ignore_index=True)
        except:
            continue
	sata.to_csv('nseOB11Nov2015.csv',index=False,mode='a')

data.sort(['symbol','Item'], ascending=[False,True])

len(data.index)/66


def get_quote(self, code, as_json=False):
    url = self.build_url_for_quote(code)
    req = Request(url, None, self.headers)
    res = self.opener.open(req)

    # Now parse the response to get the relevant data
    match = re.search(\
                r'\{<div\s+id="responseDiv"\s+style="display:none">\s+(\{.*?\{.*?\}.*?\})',
                res.read(), re.S
            )
    try:
        buffer = match.group(1)
        buffer = js_adaptor(buffer)
        response = self.clean_server_response(ast.literal_eval(buffer)['data'][0])
    except SyntaxError as err:
        raise Exception('ill formatted response')
    else:
        return self.render_response(response, as_json)
