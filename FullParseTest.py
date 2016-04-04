# Code to extract patent data from the XML files of the Full text data
import commands
from xml.dom.minidom import parse
from xml.dom import minidom
import xml.dom.minidom
with open('large_subset_full_text.xml','r') as myfile:
    inputtext = myfile.read().replace('\n', '')

    #inputtext="?xml is the best us-patent-grant"
import re
import time
start=re.escape("<?")
end=re.escape("/us-patent-grant>")
#this=re.findall(r'(?<={}).*?(?={})'.format(start, end), inputtext)
#print(this)
matchedtext= re.findall(r'(?<={}).*?(?={})'.format(start,end), inputtext)
n=len(matchedtext)
p=int
for p in range(0, n):
    matchedtext[p]= "<?"+ matchedtext[p]+ "/us-patent-grant>"
    #matchedtext[p]=matchedtext[p-2]
    #matchedtext[p]=matchedtext[-2:]
i=0
for p in matchedtext:
    time.sleep(1)
    cleanmatchedtext = matchedtext[i-2].replace("&", "and")
    i+=1
    xmldoc=xml.dom.minidom.parseString(cleanmatchedtext)
    grantdata=xmldoc.childNodes[1]
    bibdata=grantdata.childNodes[0]
    appref=bibdata.childNodes[1].attributes['appl-type'].value
    if appref == "utility":
        abstractstart = xmldoc.getElementsByTagName("abstract")
        abstract_text = abstractstart[0].getElementsByTagName("p")
        patent_title= xmldoc.getElementsByTagName("invention-title")
        patentgrant=xmldoc.getElementsByTagName("us-patent-grant")
        granted_date= patentgrant[0].attributes['date-produced'].value
        doc_number=xmldoc.getElementsByTagName('doc-number')
        claim=xmldoc.getElementsByTagName("claim-text")
        claimtext=claim[0].getElementsByTagName("claim-text")
        firstline=xmldoc.getElementsByTagName("claim-text")
        examiners=bibdata.childNodes[16]
        primary=examiners.childNodes[0]
        department=primary.childNodes[2]
        deptval=department.firstChild.nodeValue
        ClaimText=firstline[0].childNodes[0].nodeValue
        for claimtext in claimtext:
            finalclaimtext= claimtext.childNodes[0].nodeValue
            ClaimText=ClaimText + finalclaimtext
        print( doc_number[0].firstChild.nodeValue)
        print( granted_date)
        print(deptval)
        print( patent_title[0].firstChild.nodeValue)
        print(abstract_text[0].firstChild.nodeValue)
        print(ClaimText)
    else:
        print("not a utility patent")
    print("\n")
    end



