# sample script to show how to tokenize a string

import re
text = "Hello, world. Is this-- a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print (result)
