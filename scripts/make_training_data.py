"""Generates data/training.csv: synthetic labelled examples used to bootstrap the classifier.

Run `python scripts/make_training_data.py` to regenerate. Real documents corrected in the admin
interface are added to the training set on top of this seed data.
"""
import csv
import random
from pathlib import Path

PHRASES = {
    "invoice": ["invoice number", "bill to", "amount due", "payment terms net 30", "subtotal", "sales tax",
                "unit price", "quantity", "total due", "purchase order", "remit payment to", "due date",
                "description of goods", "balance outstanding", "bank transfer details", "billing address"],
    "resume": ["work experience", "education", "bachelor of science", "university", "skills", "proficient in",
               "responsible for", "led a team", "objective", "references available", "employment history",
               "certifications", "internship", "graduated with honors", "professional summary", "projects"],
    "contract": ["this agreement", "the parties", "hereinafter referred to as", "hereby agrees", "terms and conditions",
                 "limitation of liability", "indemnify", "governing law", "termination clause", "confidentiality",
                 "jurisdiction", "force majeure", "effective date", "obligations of the licensee", "witness whereof",
                 "breach of contract"],
    "research_paper": ["abstract", "we propose", "methodology", "experimental results", "dataset", "baseline models",
                       "literature review", "hypothesis", "statistical significance", "related work", "conclusion",
                       "future work", "et al", "ablation study", "evaluation metrics", "proceedings of the conference"],
    "news_article": ["officials said", "according to", "spokesperson", "yesterday", "the government announced",
                     "residents", "police reported", "election results", "stock markets", "in a statement",
                     "local authorities", "press conference", "breaking news", "city council", "witnesses described",
                     "minister told reporters"],
}
FILLER = "the of and to in for on with by at from this that was were has have will which also more other new".split()


def main(per_category=40, seed=7):
    rng = random.Random(seed)
    rows = []
    for label, phrases in PHRASES.items():
        for _ in range(per_category):
            parts = rng.sample(phrases, rng.randint(5, 9)) + rng.choices(FILLER, k=rng.randint(10, 25))
            rng.shuffle(parts)
            rows.append((" ".join(parts), label))
    rng.shuffle(rows)
    out = Path(__file__).resolve().parent.parent / "data" / "training.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "category"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
