"""Reading the XML file written next to each zarr by the microscope."""

import xml.etree.ElementTree as ET

def read_xml(path):
	"""Parse the ISO-8859-1 encoded XML file at ``path`` and return its root element.
	"""
	# Open and parse the XML file
	tree = None
	with open(path, "r", encoding="ISO-8859-1") as f:
		tree = ET.parse(f)
	return tree.getroot()

def get_xml_field(file, field):
	"""Return the text of the first element named ``field`` anywhere in the XML file ``file``.
	"""
	xml = read_xml(file)
	return xml.find(f".//{field}").text
