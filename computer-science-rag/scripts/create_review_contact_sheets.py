"""Build labelled contact sheets from rendered manual-review pages."""
from __future__ import annotations
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/pdfs/manual_review'

def sheets(prefix:str,columns:int,rows:int,width:int)->None:
    paths=sorted(SOURCE.glob(f'{prefix}_*.png'))
    per_page=columns*rows
    for offset in range(0,len(paths),per_page):
        batch=paths[offset:offset+per_page]
        thumbs=[]
        for path in batch:
            image=Image.open(path).convert('RGB')
            height=max(1,round(image.height*width/image.width))
            image.thumbnail((width,height))
            thumbs.append((path,image.copy()))
        cell_height=max((image.height for _,image in thumbs),default=1)+42
        canvas=Image.new('RGB',(columns*width,rows*cell_height),'white')
        draw=ImageDraw.Draw(canvas)
        for index,(path,image) in enumerate(thumbs):
            x=(index%columns)*width; y=(index//columns)*cell_height
            canvas.paste(image,(x,y+38))
            draw.text((x+4,y+4),path.stem,fill='black')
        canvas.save(SOURCE/f'{prefix}_contact_{offset//per_page+1:02d}.jpg',quality=88)

if __name__=='__main__':
    sheets('metadata',5,5,280)
    sheets('ocr',3,3,480)
    sheets('boundary',3,3,480)
