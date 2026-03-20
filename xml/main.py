from lxml import etree

xmlschema = etree.XMLSchema(file=str("./test.xsd"))
xml = etree.parse(str("./output.xml"))

xmlschema.assertValid(xml)
print("XML 校验通过")
