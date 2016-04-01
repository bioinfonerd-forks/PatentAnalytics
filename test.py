'''import xml.etree.ElementTree
effort  = xml.etree.ElementTree.parse('change.xml').getroot()
print(effort)'''

'''import xmltodict
with open('change.xml') as fd:
    doc = xmltodict.parse(fd.read())
print(change.xml)'''


import xml.dom.minidom
xmldoc= xml.dom.minidom.parse('fullpatenttext.xml')
main=xmldoc.childNodes
patent_title= xmldoc.getElementsByTagName("invention-title")
abstractstart=xmldoc.getElementsByTagName("abstract")
abstract_text=abstractstart[0].getElementsByTagName("p")
claim=xmldoc.getElementsByTagName("claim-text")
print(claim[0].childNodes)
print("space")


'''


claimtext=claim[0].getElementsByTagName("claim-text")
print(claimtext[0].nodeValue)


first_claim_text=claimtext[0].firstChild
print(first_claim_text)
print(first_claim_text.nodeValue)


'''


print(patent_title[0].firstChild.nodeValue)
print(abstract_text[0].firstChild.nodeValue)


