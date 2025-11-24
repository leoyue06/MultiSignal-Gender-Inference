# MultiSignal Gender Inference

This project implements a lightweight, privacy‑aware gender inference engine that estimates a user's gender only when it is missing. It uses three independent signals:

1. First Name  
2. Sport Gender Category  
3. Photo‑Based Gender Probability (Stub or Model Output)

These signals are combined through a contextual weighting system and transformed into a final probability distribution with:
- inferred gender  
- confidence score  
- per-signal attribution  
- transparent overrides  

## Repository Structure

```
/
├── docs/
│   ├── Architecture Diagram.png
│   ├── PRD.pdf
│   ├── Safety_Ethics_Considerations.pdf
│   └── README.md               # PRD README, not root README

├── examples/
│   ├── names_db.csv            # Name → gender probability lookup
│   ├── sports_db.csv           # Simple sample mapping for sports metadata
│   ├── test_1.jpg              # Demo photos for vision testing
│   ├── test_2.jpg
│   ├── test_3.jpg
│   ├── test_4.jpg
│   ├── test_5.jpg
│   └── test_6.jpg

├── prototype/
│   ├── demo.ipynb              # End-to-end demo notebook
│   ├── inference.py            # Core inference pipeline (weights + signals)
│   ├── signals.py              # Extracts name, sport, and photo signals
│   └── weights.py              # Weighting logic + contextual adjustments

├── Dockerfile                  # Optional Jupyter-based environment
├── requirements.txt            # Python dependencies
└── .gitignore

```

## Demo Video

<div align="center">
  <a href="https://www.youtube.com/watch?v=Zj2SYKlAn6o">
    <img src="https://img.youtube.com/vi/Zj2SYKlAn6o/0.jpg" width="600">
  </a>
</div>

## Summary

This is a 2–3 day interview-level project showing production-style thinking:
- clear PRD  
- safe handling of sensitive inference  
- modular architecture  
- weighted multi-signal combination  
- real attribution & interpretability  
- prototype that runs end‑to‑end
