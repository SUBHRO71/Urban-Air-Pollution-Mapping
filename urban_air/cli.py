import argparse
from .datasets import download_uci,download_india

def main():
    parser=argparse.ArgumentParser(description='Urban Air Lab local tools');parser.add_argument('action',choices=['import-uci','import-beijing','import-india','serve']);args=parser.parse_args()
    if args.action=='serve':
        import uvicorn
        uvicorn.run('urban_air.api:app',host='127.0.0.1',port=8000)
    else:
        item=download_india() if args.action=='import-india' else download_uci(args.action=='import-beijing')
        print(item['id'],item['name'],item['rows'],'rows')

if __name__=='__main__':main()
