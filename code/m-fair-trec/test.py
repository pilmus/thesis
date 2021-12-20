import jsonlines

with jsonlines.open('testfile.json', 'a') as writer:
    writer.write_all([f'{{{i}}}' for i in range(0, 10)])
