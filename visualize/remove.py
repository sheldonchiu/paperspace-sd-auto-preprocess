import os
import json

meta_file = "/Users/sheldon/Documents/booru_data/0_filter/meta_lat.json"

data = json.load(open('exclude.json','r'))
meta_lat = json.load(open(meta_file,'r'))

for file in data:
    file = os.path.splitext(file)[0]
    del meta_lat[file]
    
for file, content in meta_lat.items():
    content['caption'] = content['caption'].replace(',','')
    
    
with open(meta_file,'w') as f:
    json.dump(meta_lat,f, indent=4)