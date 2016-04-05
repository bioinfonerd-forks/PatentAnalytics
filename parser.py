import xml.dom.minidom
from config import Config
import os, re


class Patent(object):
    def __init__(self, properties):
        self.properties = properties


class Parser(object):
    def __init__(self, config):
        self.config = config

    def load_xml(self, filepath):
        with open(filepath, 'r') as myfile:
            file_text = myfile.read().replace('\n', '')
        return file_text

    def clean_xml(self, file_text):
        start = re.escape("<?")
        end = re.escape("/us-patent-grant>")
        matched_text = re.findall(r'(?<={}).*?(?={})'.format(start, end), file_text)
        clean_text = ["<?" + partition + "/us-patent-grant>" for partition in matched_text]
        clean_text = [sample.replace("&", "and") for sample in clean_text]
        return clean_text

    def parse_xml(self, input_xml):
        xmldoc = xml.dom.minidom.parseString(input_xml)
        grantdata = xmldoc.childNodes[1]
        bibdata = grantdata.childNodes[0]
        appref = bibdata.childNodes[1].attributes['appl-type'].value
        return xmldoc, appref

    def extract_patent(self, xmldoc):
        properties = dict()
        properties['title'] = xmldoc.getElementsByTagName("invention-title")[0].childNodes[0].nodeValue
        properties['date'] = xmldoc.getElementsByTagName("us-patent-grant")[0].attributes['date-produced'].value
        properties['abstract'] = xmldoc.getElementsByTagName("abstract")[0].getElementsByTagName("p")[0].childNodes[0].nodeValue
        properties['docnum'] = xmldoc.getElementsByTagName('doc-number')[0].childNodes[0].nodeValue
        properties['prefix'] = xmldoc.getElementsByTagName("claim-text")[0].childNodes[0].nodeValue
        claims = xmldoc.getElementsByTagName("claim-text")[0].getElementsByTagName("claim-text")
        properties['claims'] = [claim.childNodes[0].nodeValue for claim in claims]
        patent = Patent(properties)
        return patent

    def import_data(self, filename):
        filepath = os.path.join(self.config.data_dir, filename)
        file_text = self.load_xml(filepath)
        clean_text = self.clean_xml(file_text)
        patents = []
        for partition in clean_text:
            xmldoc, appref = self.parse_xml(partition)
            if appref == 'utility':
                patents.append(self.extract_patent(xmldoc))
        return patents

if __name__ == "__main__":
    config = Config()
    parser = Parser(config)
    patents = parser.import_data('ipg160105.xml')
    pass
