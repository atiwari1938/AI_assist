import argparse
from retriever import retrieve
from generator import generate_response

def main():
    p = argparse.ArgumentParser(description="SupportAssist AI CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("retrieve")
    q.add_argument("--query", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--query", required=True)

    args = p.parse_args()
    if args.cmd == "retrieve":
        print(retrieve(args.query))
    elif args.cmd == "generate":
        print(generate_response(args.query, retrieve(args.query)))

if __name__ == "__main__":
    main()
