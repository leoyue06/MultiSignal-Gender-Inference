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
/docs
    PRD.pdf
    Architecture Diagram.png
    Safety_Ethics_Considerations.pdf
    README.md

/prototype
    inference.py
    signals.py
    weights.py
    demo.ipynb

/examples
    names_sample.csv
    dummy_image.jpg
```

## Summary

This is a 2–3 day interview-level project showing production-style thinking:
- clear PRD  
- safe handling of sensitive inference  
- modular architecture  
- weighted multi-signal combination  
- real attribution & interpretability  
- prototype that runs end‑to‑end  
