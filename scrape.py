from urllib.request import Request, urlopen
from lxml.html import fromstring
import pandas as pd
from datetime import datetime
import os.path
import time

def floaterr(string):
    """
    convert any not numerical/floating point error to 0.0, a float
    """
    try:
        floatnum = float(string)
    except:
        floatnum = 0
    return floatnum

def optionchain(exdate = '25FEB2016'):
    """
    Actual scraping done here
    """
    ocurl = "http://www.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?segmentLink=17&instrument=OPTIDX&symbol=NIFTY&date="+exdate
    headers = {'Accept' : '*/*',
               'Accept-Language' : 'en-US,en;q=0.5',
               'Host': 'nseindia.com',
               'Referer': 'http://www.nseindia.com/live_market/dynaContent/live_market.htm',
               'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1;WOW64;rv:28.0) Gecko Firefox/35',
               'X-Requested-With': 'XMLHttpRequest'}

    req = Request(ocurl, None, headers)
    response = urlopen(req)
    the_page = response.read() # works upto this; got the page in a string
    ptree = fromstring(the_page)
    tr_nodes = ptree.xpath('//table[@id="octable"]//tr')[1:]
    td_content = [[td.text_content().strip() for td in tr.xpath('td')] for tr in tr_nodes[1:]]
    td_content = [[string.replace(',', '') for string in sublist] for sublist in td_content]
    td_content = [[floaterr(string) for string in sublist] for sublist in td_content]
    return td_content[:-1]

def get_ocdata(exdate = '25FEB2016', start = 540*60, stop = 930*60, waitflag= True, nosleep= True):
    """
    start: Market open time/when the data scraping starts, in seconds elapsed from 12:00 AM
    stop:  Market closing time/when the data scraping stops, in seconds elapsed from 12:00 AM
    waitflag: whether to wait for the time specified in 'start' variable to occur, sleep till then
    nosleep: whether to scrape continually or give 1 sec pause between scrapes,
             will sleep if set to True if scraped before 1 sec interval
             if scrapes take more than 1 sec has no effect
             good to use if too much data traffic/network overloaded
    """

    currtime = time.localtime()
    dct = time.strftime('%d%b%Y', currtime).upper()
    datafile = 'nseOC' + dct + '.csv'
    if not os.path.isfile(datafile):
        dfcolnames = ["OI.call", "changeOI.call", "volume.call", "IV.call", "LTP.call", "netChange.call", "bidQ.call", "bidP.call", "askP.call", "askQ.call", "strike", "bidQ.put", "bidP.put", "askP.put", "askQ.put", "netChange.put", "LTP.put", "IV.put", "volume.put", "changeOI.put", "OI.put", "timestamp"]
        dfcolnames = pd.DataFrame(dfcolnames)
        dfcolnames = pd.DataFrame(dfcolnames).T
        dfcolnames.to_csv(datafile, index=False, mode='a', header=False)

    ptime = datetime.now()

    if waitflag == True:
        secs = (ptime.hour*60*60 + ptime.minute*60 + ptime.second)
        if secs < start:
            time.sleep(stop - secs)

    while True:
        if (ptime.hour*60*60 + ptime.minute*60 + ptime.second) > stop:
            break
        else:
            try:
                ptime = datetime.now()
                scrapedata = pd.DataFrame(optionchain(exdate))
                scrapedata = scrapedata.ix[:, 1:21]
                #scrapedata.columns = dfcolnames
                scrapedata['timestamp'] = ptime
                scrapedata.to_csv(datafile, index=False, mode='a', header=False)
                if nosleep == False:
                    end = datetime.now()
                    tdiff = end-ptime
                    time.sleep(1 - (tdiff.seconds + tdiff.microseconds*1.0/10**6))
            except:
                continue
    return 0

get_ocdata(stop = 930*60, nosleep = False)
