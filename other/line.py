import re
from pprint import pprint
line = r" 1290 (_sex  male) (_age  child) (_race white) (_face smiling) (_prop '(glasses ))"

match = re.findall(r"(\d+)|\b_missing.descriptor|\b_(?:sex|age|[rf]ace)\s+(\w+)|_prop..\(([^()]*).\)\)",
                   line)
labelList = list(["".join(x) for x in match])

if len(labelList) == 2:
    labelList.append(labelList[1])

pprint(match)