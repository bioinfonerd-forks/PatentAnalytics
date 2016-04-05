import xml.dom.minidom
from config import Config
import os

class Patent(object):
    def __init__(self, properties):
        self.properties = properties


class Parser(object):
    def __init__(self, config):
        self.config = config

    def import_xml(self, filename):
        filepath = os.path.join(config.data_dir, filename)
        xmldoc = xml.dom.minidom.parse(filepath)
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
    config = Config()
    parser = Parser(config)
    properties = parser.import_xml('fullpatenttext.xml')
    patent = Patent(properties)
