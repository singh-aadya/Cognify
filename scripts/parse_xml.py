//
import xml.etree.ElementTree as ET

def parse_opus_nllb_xml(xml_file, en_out, hi_out):
    """
    Parse OPUS NLLB XML file and extract parallel English-Hindi sentences.
    Save them line-by-line in two separate files.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    en_sentences = []
    hi_sentences = []

    for doc in root.findall('doc'):
        for seg in doc.findall('seg'):
            en_text = seg.find('en').text
            hi_text = seg.find('hi').text
            if en_text and hi_text:
                en_sentences.append(en_text.strip())
                hi_sentences.append(hi_text.strip())

    with open(en_out, 'w', encoding='utf-8') as f_en, open(hi_out, 'w', encoding='utf-8') as f_hi:
        f_en.write('\n'.join(en_sentences))
        f_hi.write('\n'.join(hi_sentences))

    print(f"Extracted {len(en_sentences)} sentence pairs to {en_out} and {hi_out}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python parse_xml.py <input_xml> <output_en.txt> <output_hi.txt>")
        exit(1)
    parse_opus_nllb_xml(sys.argv[1], sys.argv[2], sys.argv[3])
