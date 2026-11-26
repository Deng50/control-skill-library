"""Find task-specific skills and their checked local engineering evidence."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def lookup(query):
    index=json.loads((ROOT/"index.json").read_text(encoding="utf-8"))
    query=query.lower()
    matches=[]
    for item in index:
        hits=[term for term in item["terms"] if term.lower() in query]
        if hits:
            matches.append((sum(len(t) for t in hits),item["skill"],item))
    return [item for _,_,item in sorted(matches,key=lambda m:(-m[0],m[1]))]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--limit",type=int,default=3)
    a=p.parse_args()
    rows=lookup(a.query)
    if not rows:
        p.exit(1,"No term matched. Browse index.json by role and task.\n")
    print(json.dumps(rows[:a.limit],indent=2,ensure_ascii=False))
