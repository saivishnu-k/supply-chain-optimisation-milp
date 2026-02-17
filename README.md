# Supply Chain Optimisation using MILP and Stochastic Programming
GreenGlow Cosmetics

## Overview
This repository contains the final report and supporting optimisation code for a multi echelon global supply chain problem for GreenGlow Cosmetics. The model balances three competing objectives across suppliers, production plants, and regional markets:
cost minimisation, CO2 emissions reduction, and service level performance.

The solution uses a weighted multi objective Mixed Integer Linear Programming formulation, extended with a stochastic programming approach to account for uncertainty in supplier deliveries and regional demand.

## Supply chain setting
Network
4 suppliers
4 production plants (Europe, North America, Asia, South America)
6 demand regions (Asia, Europe, North America, South America, Middle East, Africa)
3 finished products (A, B, C)

Uncertainty
Supplier delivery variation approximately plus or minus 10 to 20 percent
Regional demand variation approximately plus or minus 10 to 15 percent

## Model
Decision variables
X_spm raw material flow from supplier s to plant p by mode m
Y_pk production quantity of product k at plant p
Z_pkd distribution quantity from plant p to region d
S_kd shortage quantity for product k in region d

Objective
Weighted sum of three components with weights:
cost 0.3
emissions 0.3
shortage penalties 0.4

Constraints
supplier capacity and mode availability
raw material requirements linked to production
plant capacity
flow conservation between production and shipments
demand satisfaction with explicit shortage variables
minimum service level constraint of 12 percent for each product and region

Stochastic extension
Five scenarios with equal probability
First stage binary plant expansion decisions
Second stage operational decisions per scenario

## Key findings
No plant expansion is recommended across deterministic, stochastic, and sensitivity tests.
Asia is the primary production hub, around 43 percent of total production, driven by low operating cost and strong emissions performance.
The model recommends differentiated service levels:
prioritise higher service in high penalty regions (Middle East and Africa)
accept larger strategic shortages in lower penalty regions (Asia and North America).
Transportation mode selection is stable by supplier:
sea for Suppliers 1 and 3
air for Supplier 2
mixed strategy for Supplier 4 depending on conditions.

## Files
GreenGlow_Supply_Chain_Optimisation_Report.pdf
Final academic report

analysis.py
Optimisation model implementation in Python using PuLP

## Data notice
The original dataset is not included. The code is provided as a reference implementation of the modelling approach and can be adapted to similar supply chain datasets.

## Skills demonstrated
Prescriptive analytics
MILP optimisation modelling
Stochastic programming concepts
Supply chain network design
Sustainability analytics and trade off analysis
Python and PuLP

## Author
Sai Vishnu Kandagattla
MSc Business Analytics
University College Cork
