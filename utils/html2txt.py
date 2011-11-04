"""
========================================================
html2txt - single hard-coded url test
========================================================

Refined version of html2txt:
1) Parsing html using lxml
2) Well-handled domain name extraction
3) In-depth cleanup that cleans all meaningless tabs, spaces, empytlines
4) Clean-up lines that are shorter than a given length (line 82)
5) Cutomized input (.txt file that contains url to the target .html files, one url each line)
"""

from __future__ import with_statement
from urlparse import urlparse

import os
import re
import StringIO
import lxml.html

from urllib2 import urlopen

# print __doc__

#################################################################################
# Auxiliary function and file for correctly extract domain from url
# Author:
# http://stackoverflow.com/questions/1066933/python-extract-domain-name-from-url
# load tlds, ignore comments and empty lines:
with open("utils/effective_tld_names.dat.txt") as tldFile:
    tlds = [line.strip() for line in tldFile if line[0] not in "/\n"]

def getDomain(url, tlds):
    urlElements = urlparse(url)[1].split('.')
    # urlElements = ["abcde","co","uk"]

    for i in range(-len(urlElements),0):
        lastIElements = urlElements[i:]
        #    i=-3: ["abcde","co","uk"]
        #    i=-2: ["co","uk"]
        #    i=-1: ["uk"] etc

        candidate = ".".join(lastIElements) # abcde.co.uk, co.uk, uk
        wildcardCandidate = ".".join(["*"]+lastIElements[1:]) # *.co.uk, *.uk, *
        exceptionCandidate = "!"+candidate

        # match tlds: 
        if (exceptionCandidate in tlds):
            return ".".join(urlElements[i:]) 
        if (candidate in tlds or wildcardCandidate in tlds):
            return ".".join(urlElements[i-1:])
            # returns "abcde.co.uk"

    raise ValueError("Domain not in global list of TLDs")
# print getDomain("http://abcde.co.uk",tlds)


# Function for html2txt using lxml
# Author:
# http://groups.google.com/group/cn.bbs.comp.lang.python/browse_thread/thread/781a357e2ce66ce8
def html2text(html):
    tree = lxml.etree.fromstring(html, lxml.etree.HTMLParser()) if isinstance(html, basestring) else html 
    for skiptag in ('//script', '//iframe', '//style', 
                    '//link', '//meta', '//noscript', '//option'):    
        for node in tree.xpath(skiptag):
            node.getparent().remove(node)
    # return lxml.etree.tounicode(tree, method='text')
    return lxml.etree.tostring(tree, encoding=unicode, method='text')


#Function for cleanup the text:
# 1: clearnup: 1)tabs, 2)spaces, 3)empty lines;
# 2: remove short lines
def textcleanup(text, length_threshold=35):
    # temp list for process
    text_list = []
    for s in text.splitlines():
        # Strip out meaningless spaces and tabs
        s = s.strip()
        # New improvement: removing all extra spaces
        s = re.sub(r' +', ' ', s)        
        # Set length limit
        if s.__len__() > length_threshold:
            text_list.append(s)
    cleaned = os.linesep.join(text_list)
    # Get rid of empty lines
    cleaned = os.linesep.join([s for s in cleaned.splitlines() if s])
    return cleaned


class Html2txtConverter:

    def __init__(self, url='', length_threshold=35):
        self.url = url
        self.length_threshold = length_threshold

    def get_paragraphs(self):

        # declear an empty list for the paragraphs
        paragraphs = []
        
        # read the url
        line = self.url

        # get the domain from url for naming of the txt file
        domain = getDomain(line, tlds)
        
        # parse to get the parsed tree
        htmltree = lxml.html.parse(urlopen(line))

        # html2txt
        # Ignore any characte that is not ascii coded
        output = html2text(htmltree).encode('ascii', 'ignore')

        # cleanup - str with multiple lines
        cleaned = textcleanup(output, self.length_threshold)

        # store back to the list
        paragraphs = cleaned.split('\n')

        return paragraphs