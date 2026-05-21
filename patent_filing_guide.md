# Indian Patent Office — Provisional Application Guide
# FedAdapt System — Patents Act 1970

---

## What is a Provisional Application?
- Locks in your **priority date** immediately
- Costs only ₹1,750 (individual) — or ₹750 with DPIIT
- Gives you 12 months to file the Complete Specification
- You can write "Patent Pending" after filing

---

## Step 1: DPIIT Startup India Registration (80% fee reduction)

Even as an individual researcher this applies to you.

1. Go to: https://www.startupindia.gov.in
2. Click "Register" → "Startup"
3. Fill your details as an individual innovator
4. Upload: Aadhaar card, PAN card
5. You receive a DPIIT recognition certificate
6. This reduces patent fees by 80%
   - Filing fee: ₹1,750 → ₹350
   - Examination fee: ₹4,000 → ₹800

---

## Step 2: Create IPO Account

1. Go to: https://ipindiaonline.gov.in
2. Click "e-filing" → "New User Registration"
3. Register as: Individual Applicant
4. Verify email and login

---

## Step 3: Fill Form 1 (Application for Grant of Patent)

**Title of Invention:**
Adaptive Federated Learning System with Heterogeneity-Aware
Aggregation and Layered Differential Privacy for Edge Devices
**Applicant Details:**
- Name: [Your full name]
- Nationality: Indian
- Address: [Your address], Bengaluru, Karnataka
- Category: Individual (tick this for reduced fee)

**Inventor Details:** (same as applicant if you are the inventor)

---

## Step 4: Write the Provisional Specification

This is the core document. It does NOT need claims yet.
Paste this as your provisional specification:

---

**TITLE:** Adaptive Federated Learning System with 
Heterogeneity-Aware Aggregation and Layered Differential 
Privacy for Edge Devices

**FIELD OF INVENTION:**
This invention relates to distributed machine learning systems,
specifically to federated learning methods that operate on 
heterogeneous edge devices with formal differential privacy 
guarantees.

**BACKGROUND:**
Existing federated learning systems (FedAvg, McMahan et al. 2017)
have two limitations: (1) they assume homogeneous client devices
and (2) they apply uniform privacy noise to all model layers,
degrading model utility unnecessarily.

**SUMMARY OF INVENTION:**
The present invention provides a system and method comprising
three novel technical components:

1. Dynamic Compression Ratio Selector — adapts gradient
   compression per device per round based on measured compute
   capacity.

2. Heterogeneity-Aware Reliability-Weighted Aggregation —
   computes reliability scores from client accuracy history
   and weights aggregation by both sample count and reliability.

3. Layered Differential Privacy Budget Allocator — classifies
   neural network layers into sensitivity tiers and applies
   differentiated Gaussian noise proportional to each tier.

**TECHNICAL EFFECT:**
Experimental validation on MNIST benchmark achieved 99.07%
accuracy across 3 client devices in 5 federated rounds under
a (10.0, 1e-5)-differential privacy guarantee, on commodity
hardware without GPU acceleration.

**BRIEF DESCRIPTION OF DRAWINGS:**
Figure 1: Four-panel chart showing (a) global accuracy per
round, (b) loss convergence, (c) global vs client accuracy
comparison, and (d) privacy budget consumption over rounds.

---

## Step 5: Attach Documents

Upload these files:
- [ ] Form 1 (filled online)
- [ ] Provisional Specification (text above)
- [ ] patent_evidence_figure1.png (labeled as Figure 1)
- [ ] patent_evidence_report.txt (as Annexure A)
- [ ] Fee payment receipt

---

## Step 6: Pay Filing Fee

- Individual (no DPIIT): ₹1,750
- Individual (with DPIIT): ₹350
- Pay online via net banking / UPI at ipindiaonline.gov.in

---

## Step 7: After Filing

You will receive:
- **Application Number** — keep this safe
- **Priority Date** — the date that matters legally
- You can now write **"Patent Pending"** on your work

Within 12 months you must file:
- Complete Specification with full claims
- Form 2

Within 48 months you must file:
- Request for Examination (Form 18)
- Fee: ₹4,000 (individual) or ₹800 (with DPIIT)

---

## Recommended Patent Agent (optional but helpful)

A registered patent agent drafts stronger claims.
Find agents at: https://ipindia.gov.in/patent-agent.htm
Typical cost: ₹15,000–₹30,000 for full filing assistance

For provisional filing alone, you can file yourself
using the documents already generated in this project.

---

## Timeline Summary

| Action | Deadline | Cost |
|--------|----------|------|
| File Provisional | ASAP | ₹350–₹1,750 |
| File Complete Spec | Within 12 months | ₹350–₹1,750 |
| Request Examination | Within 48 months | ₹800–₹4,000 |
| Patent Granted | ~3-5 years | — |