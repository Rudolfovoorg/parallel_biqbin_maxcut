import json
from glob import glob
import os
types = {
    'gw': 'GW',
    'sa': 'SA',
    'qa': 'QA'
}

if __name__ == "__main__":
    for t in types:
        print(t, types[t])
        for filename in glob(f'/mnt/c/Users/Beno/Downloads/{types[t]}/{types[t]}/*'):
            with open(filename, 'r') as fr:
                sol_data = json.load(fr)
            
            
        
        
        # print(filename)