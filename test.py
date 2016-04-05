# Code to extract patent data from the XML files of the Full text data
import xml.dom.minidom
xmldoc= xml.dom.minidom.parse('fullpatenttext.xml')
main=xmldoc.childNodes
patent_title= xmldoc.getElementsByTagName("invention-title")
patentgrant=xmldoc.getElementsByTagName("us-patent-grant")
granted_date= patentgrant[0].attributes['date-produced'].value
abstractstart=xmldoc.getElementsByTagName("abstract")
abstract_text=abstractstart[0].getElementsByTagName("p")
doc_number=xmldoc.getElementsByTagName('doc-number')
claim=xmldoc.getElementsByTagName("claim-text")
claimtext=claim[0].getElementsByTagName("claim-text")
firstline=xmldoc.getElementsByTagName("claim-text")
ClaimText=firstline[0].childNodes[0].nodeValue
for claimtext in claimtext:
    finalclaimtext=first_claim_text = claimtext.childNodes[0].nodeValue
    ClaimText=ClaimText + finalclaimtext


print(doc_number[1].firstChild.nodeValue)
print(granted_date)
print(patent_title[0].firstChild.nodeValue)
print(abstract_text[0].firstChild.nodeValue)
print(ClaimText)

