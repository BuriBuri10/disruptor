# This script is a conceptual model for the United Airlines Disruption Management system.
# It uses the 'pulp' library to outline the structure of a Mixed-Integer Linear Programming (MILP) model.
# NOTE: This is for demonstration and requires real data and a solver to be operational.

import pulp
import pandas as pd

# --- 1. Problem Setup & Data Simulation ---
# In a real system, this data would be streamed from United's operational databases.

# Mock Flights Data
flights_data = {
    'FlightID': ['UA482', 'UA1992', 'UA211', 'UA635'],
    'Origin': ['EWR', 'ORD', 'SFO', 'EWR'],
    'Destination': ['SFO', 'LAX', 'EWR', 'ORD'],
    'DepartureTime': [800, 830, 900, 915], # In minutes from midnight
    'ArrivalTime': [1100, 1100, 1700, 1100],
    'AircraftType': ['B777', 'B739', 'B78X', 'A320'],
    'Passengers': [250, 179, 300, 150],
    'HighValuePax': [20, 15, 35, 10] # e.g., Global Services / 1K members
}
flights = pd.DataFrame(flights_data).set_index('FlightID')

# Mock Aircraft Data
aircraft_data = {
    'TailNum': ['N770UA', 'N771UA', 'N730UA', 'N320UA'],
    'Type': ['B777', 'B78X', 'B739', 'A320'],
    'Location': ['EWR', 'SFO', 'ORD', 'EWR']
}
aircraft = pd.DataFrame(aircraft_data).set_index('TailNum')

# Disruption Scenario: EWR has reduced capacity. Some flights must be delayed or canceled.
# Cost Parameters (Illustrative)
COST_CANCEL_FLIGHT = 50000  # Base cost for canceling a flight
COST_DELAY_MINUTE = 100     # Cost per minute of delay
COST_AFFECTED_PAX = 200     # Cost per standard passenger on a canceled flight
COST_AFFECTED_HV_PAX = 2500 # **KEY**: Higher cost for impacting a high-value passenger

# --- 2. Model Initialization ---
# Define the optimization problem
model = pulp.LpProblem("Disruption_Recovery_Model", pulp.LpMinimize)

# --- 3. Decision Variables ---

# a) Flight Status Variables (Binary)
# x_f = 1 if flight f is canceled, 0 otherwise
cancel_vars = pulp.LpVariable.dicts("IsCanceled", flights.index, cat='Binary')

# b) Delay Variables (Continuous, Non-negative)
# d_f = delay in minutes for flight f
delay_vars = pulp.LpVariable.dicts("Delay", flights.index, lowBound=0, cat='Continuous')

# c) Aircraft Assignment Variables (Binary)
# a_f_t = 1 if flight f is assigned to aircraft tail number t, 0 otherwise
# This creates a variable for each valid flight-aircraft combination
assignment_vars = pulp.LpVariable.dicts("Assign",
                                       [(f, t) for f in flights.index for t in aircraft.index if flights.loc[f, 'AircraftType'] == aircraft.loc[t, 'Type']],
                                       cat='Binary')

# --- 4. Objective Function ---
# The goal is to minimize the total weighted cost of the recovery plan.

# a) Cost of Cancellations
cancellation_cost = pulp.lpSum(
    cancel_vars[f] * (
        COST_CANCEL_FLIGHT +
        flights.loc[f, 'Passengers'] * COST_AFFECTED_PAX +
        flights.loc[f, 'HighValuePax'] * COST_AFFECTED_HV_PAX
    ) for f in flights.index
)

# b) Cost of Delays
delay_cost = pulp.lpSum(delay_vars[f] * COST_DELAY_MINUTE for f in flights.index)

# c) Combine costs into the final objective function
model += cancellation_cost + delay_cost, "Total_Disruption_Cost"


# --- 5. Constraints ---
# These are the rules that the final solution must obey.

# a) Flight Coverage Constraint:
# Every flight must either be operated or canceled.
# An operated flight must be assigned to exactly one aircraft.
for f in flights.index:
    model += pulp.lpSum(assignment_vars[f, t] for t in aircraft.index if (f, t) in assignment_vars) == (1 - cancel_vars[f]), f"Flight_Coverage_{f}"

# b) Aircraft Utilization Constraint:
# Each aircraft can be assigned to at most one flight in this simple model.
# A real model would handle multi-leg routings.
for t in aircraft.index:
    model += pulp.lpSum(assignment_vars[f, t] for f in flights.index if (f, t) in assignment_vars) <= 1, f"Aircraft_Utilization_{t}"

# c) Aircraft Location Constraint (Flow Balance):
# An aircraft must be at the origin airport of the flight it's assigned to.
# This is a simplified representation of network flow.
for f in flights.index:
    for t in aircraft.index:
        if (f, t) in assignment_vars:
            # If aircraft is not at the origin, the assignment is impossible.
            if aircraft.loc[t, 'Location'] != flights.loc[f, 'Origin']:
                model += assignment_vars[f, t] == 0, f"Aircraft_Location_{f}_{t}"

# d) Disruption-Specific Constraint (Example):
# Simulate reduced capacity at EWR. Let's say a maximum of 1 flight can depart on time.
# This requires more complex time-window constraints in a real model, but we can simulate it.
ewr_flights = flights[flights['Origin'] == 'EWR'].index
# Forcing a delay or cancellation on all but one EWR flight
# This is a simplified way to model a capacity constraint.
model += pulp.lpSum(cancel_vars[f] for f in ewr_flights) + pulp.lpSum(delay_vars[f] for f in ewr_flights) >= 1, "EWR_Capacity_Constraint"


# --- 6. Solve the Model ---
# In a real scenario, you would uncomment the line below and use a powerful solver.
# model.solve(pulp.GUROBI_CMD()) # Or another solver like CPLEX, CBC

# For this demo, we can print the model structure.
print("--- Model Formulation ---")
print(model)

# --- 7. Interpret the Results (Simulated) ---
# After solving, you would parse the variable values to get the recovery plan.
print("\n--- Simulated Optimal Recovery Plan ---")
# Example of what the output would look like:
print("Flight UA482 (EWR-SFO): Delayed by 60 minutes.")
print("Flight UA635 (EWR-ORD): Canceled. (Model chose this due to fewer high-value pax)")
print("Flight UA1992 (ORD-LAX): On Time.")
print("Flight UA211 (SFO-EWR): On Time.")
print("\nTotal Estimated Cost: $156,500")