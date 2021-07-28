import csv
import os
from xml.dom.minidom import parse


def parser_xml_to_csv(filename):
    print("Transfer xml to csv :" + filename)
    file_path = filename + '.xml'
    if os.path.exists(file_path):
        return
    dom_tree = parse(file_path)

    root_node = dom_tree.documentElement
    csv_writer = csv.writer(open(filename + '.csv', 'w'))
    csv_writer.writerow(['filename', 'conversation_no', 'act_tag', 'caller', 'text', 'pos'])
    dialogues = root_node.getElementsByTagName("dialogue")
    for dialogue in dialogues:
        conversation_no = dialogue.getAttribute("name")
        if not conversation_no:
            continue
        for turn in dialogue.getElementsByTagName("turn"):
            for utt in turn.getElementsByTagName("utt"):
                text_node = utt.firstChild
                try:
                    data = text_node.data
                except Exception as e:
                    data = ''

                row = [filename, conversation_no, utt.getAttribute("da"), turn.getAttribute("speaker"), data,
                       data.replace(' ', '/')]
                csv_writer.writerow(row)
    print("Finished transfer xml to csv :" + file_path)
