import xml.dom.minidom


class Patent(object):
    def __init__(self, properties):
        self.properties = properties


class Parser(object):
    def __init__(self, config):
        self.config = config

    def import_xml(self, filename):
        xmldoc = xml.dom.minidom.parse(filename)
        properties = dict()
        properties['title'] = xmldoc.getElementsByTagName("invention-title")
        properties['date'] = xmldoc.getElementsByTagName("us-patent-grant")[0].attributes['date-produced'].value
        properties['abstract'] = xmldoc.getElementsByTagName("abstract")[0].getElementsByTagName("p")
        properties['refnum'] = xmldoc.getElementsByTagName('doc-number')
        properties['prefix'] = xmldoc.getElementsByTagName("claim-text")[0].childNodes[0].nodeValue
        claims = xmldoc.getElementsByTagName("claim-text")[0].getElementsByTagName("claim-text")
        properties['claims'] = [claim.childNodes[0].nodeValue for claim in claims]
        return properties

if __name__ == "__main__":
    parser = Parser([])
    properties = parser.import_xml('fullpatenttext.xml')
    patent = Patent(properties)
