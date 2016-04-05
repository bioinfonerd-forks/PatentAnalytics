import xml.dom.minidom
from config import Config
import os, re


class Patent(object):
    def __init__(self, properties):
        self.properties = properties


class Parser(object):
    def __init__(self, config):
        self.config = config

    def clean_xml(self, filename):
        filepath = os.path.join(config.data_dir, filename)
        with open(filepath,'r') as myfile:
            inputtext = myfile.read().replace('\n', '')
        start=re.escape("<?")
        end=re.escape("/us-patent-grant>")
        matchedtext= re.findall(r'(?<={}).*?(?={})'.format(start,end), inputtext)
        clean_text = ["<?" + partition + "/us-patent-grant>" for partition in matchedtext]
        return clean_text

    def import_xml(self, filename):
        filepath = os.path.join(config.data_dir, filename)
        xmldoc = xml.dom.minidom.parse(filepath)
        properties = dict()
        properties['title'] = xmldoc.getElementsByTagName("invention-title")[0].childNodes[0].nodeValue
        properties['date'] = xmldoc.getElementsByTagName("us-patent-grant")[0].attributes['date-produced'].value
        properties['abstract'] = xmldoc.getElementsByTagName("abstract")[0].getElementsByTagName("p")[0].childNodes[0].nodeValue
        properties['refnum'] = xmldoc.getElementsByTagName('application-reference')[0].childNodes[0].nodeValue
        properties['prefix'] = xmldoc.getElementsByTagName("claim-text")[0].childNodes[0].nodeValue
        claims = xmldoc.getElementsByTagName("claim-text")[0].getElementsByTagName("claim-text")
        properties['claims'] = [claim.childNodes[0].nodeValue for claim in claims]
        return properties

if __name__ == "__main__":
    config = Config()
    parser = Parser(config)
    properties = parser.import_xml('ipg160105.xml')
    patent = Patent(properties)
