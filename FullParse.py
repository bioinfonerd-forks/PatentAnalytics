# Code to extract patent data from the XML files of the Full text data
import os
import sys

from xml.dom.minidom import parse
import xml.dom.minidom
import re
import time
i=0
doc=[]
date=[]
artunit=[]
title=[]
abstracttext=[]
firstclaim=[]

with open('data/ipg160105.xml','r') as myfile:
    inputtext = myfile.read().replace('\n', '')
start=re.escape("<?")
end=re.escape("/us-patent-grant>")
matchedtext= re.findall(r'(?<={}).*?(?={})'.format(start,end), inputtext)
n=len(matchedtext)
p=inta
for p in range(0, n):

    matchedtext[p]= "<?"+ matchedtext[p]+ "/us-patent-grant>"

for p in matchedtext:
    try:
        time.sleep(0.05)
        cleanmatchedtext = matchedtext[i-2].replace("&", "and")
        xmldoc=xml.dom.minidom.parseString(cleanmatchedtext)
        grantdata=xmldoc.childNodes[1]
        bibdata=grantdata.childNodes[0]
        appref=bibdata.childNodes[1].attributes['appl-type'].value
        if appref == "utility":

                abstractstart = xmldoc.getElementsByTagName("abstract")
                abstract_text = abstractstart[0].getElementsByTagName("p")
                patent_title = xmldoc.getElementsByTagName("invention-title")
                patentgrant = xmldoc.getElementsByTagName("us-patent-grant")
                granted_date = patentgrant[0].attributes['date-produced'].value
                doc_number = xmldoc.getElementsByTagName('doc-number')
                claim = xmldoc.getElementsByTagName("claim-text")
                claimtext = claim[0].getElementsByTagName("claim-text")
                firstline = xmldoc.getElementsByTagName("claim-text")
                apple = xmldoc.getElementsByTagName("department")
                deptval = apple[0].childNodes[0].nodeValue
                ClaimText = firstline[0].childNodes[0].nodeValue
                for claimtext in claimtext:
                    finalclaimtext = claimtext.childNodes[0].nodeValue
                    ClaimText = ClaimText + finalclaimtext
                print(doc_number[1].firstChild.nodeValue)
                print(granted_date)
                print(deptval)
                print(patent_title[0].firstChild.nodeValue)
                print(abstract_text[0].firstChild.nodeValue)
                print(ClaimText)
                doc.append(doc_number[0].firstChild.nodeValue)
                date.append(granted_date)
                artunit.append(deptval)
                title.append(patent_title[0].firstChild.nodeValue)
                abstracttext.append(abstract_text[0].firstChild.nodeValue)
                firstclaim.append(ClaimText)
        else:
            print("not a utility patent")
    except:
        pass
    i += 1
    print(i)
    print(n)
    print("\n")
    end

print(doc)
print(date)
print(artunit)
print(title)
print(abstracttext)
print(firstclaim)
