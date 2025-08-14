import streamlit as st
import pandas as pd
import numpy as np
import pulp
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="United Airlines Disruption Management",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions & Data Setup ---

@st.cache_data
def get_initial_data():
    """
    Creates the initial flight and aircraft schedules with more realistic and extensive data.
    This data serves as the input for the MILP model.
    """
    flights_data = {
        'FlightID': [
            'UA1989', 'UA2428', 'UA635', 'UA1722', 'UA511', 'UA2402', 'UA331', 'UA482', 'UA1992', 'UA211',
            'UA1118', 'UA123', 'UA789', 'UA2001', 'UA555', 'UA300', 'UA1812', 'UA950', 'UA401', 'UA1602'
        ],
        'Origin': [
            'ORD', 'EWR', 'EWR', 'IAH', 'LAX', 'EWR', 'DEN', 'EWR', 'ORD', 'SFO',
            'DEN', 'SFO', 'LAX', 'ORD', 'IAH', 'EWR', 'SFO', 'LAX', 'ORD', 'IAD'
        ],
        'Destination': [
            'EWR', 'MIA', 'ORD', 'DEN', 'ORD', 'FLL', 'IAH', 'SFO', 'LAX', 'EWR',
            'LAX', 'IAD', 'JFK', 'BOS', 'MCO', 'SFO', 'DEN', 'EWR', 'MIA', 'LAX'
        ],
        'AircraftType': [
            'B77W', 'A320', 'A320', 'B738', 'B752', 'A319', 'B739', 'B777', 'B739', 'B78X',
            'B738', 'B777', 'A320', 'B739', 'B738', 'B78X', 'A319', 'B77W', 'B752', 'B739'
        ],
        'Departure_Orig': pd.to_datetime([
            '2025-08-15 07:00', '2025-08-15 07:15', '2025-08-15 08:15', '2025-08-15 08:30', '2025-08-15 08:45',
            '2025-08-15 09:00', '2025-08-15 09:20', '2025-08-15 09:45', '2025-08-15 10:00', '2025-08-15 10:30',
            '2025-08-15 11:00', '2025-08-15 11:25', '2025-08-15 12:00', '2025-08-15 12:15', '2025-08-15 12:45',
            '2025-08-15 13:00', '2025-08-15 13:30', '2025-08-15 14:00', '2025-08-15 14:20', '2025-08-15 15:00'
        ]),
        'Arrival_Orig': pd.to_datetime([
            '2025-08-15 10:30', '2025-08-15 10:15', '2025-08-15 10:00', '2025-08-15 10:00', '2025-08-15 14:15',
            '2025-08-15 12:00', '2025-08-15 12:30', '2025-08-15 12:45', '2025-08-15 13:00', '2025-08-15 18:30',
            '2025-08-15 13:00', '2025-08-15 19:30', '2025-08-15 20:15', '2025-08-15 14:30', '2025-08-15 15:15',
            '2025-08-15 16:00', '2025-08-15 15:30', '2025-08-15 22:00', '2025-08-15 17:00', '2025-08-15 17:30'
        ]),
        'Passengers': [
            350, 150, 150, 166, 176, 128, 179, 250, 179, 300,
            166, 250, 150, 179, 166, 300, 128, 350, 176, 179
        ],
        'HighValuePax': [
            40, 10, 9, 12, 18, 8, 14, 20, 15, 35,
            11, 22, 13, 19, 15, 33, 7, 45, 16, 21
        ],
    }
    flights_df = pd.DataFrame(flights_data).set_index('FlightID')

    aircraft_data = {
        'TailNum': [
            'N777UA', 'N320UA', 'N321UA', 'N738UA', 'N752UA', 'N319UA', 'N731UA', 'N770UA', 'N730UA', 'N787UA',
            'N739UA', 'N771UA', 'N322UA', 'N732UA', 'N733UA', 'N788UA', 'N318UA', 'N772UA', 'N753UA', 'N734UA',
            'N735UA_Reserve', 'N325UA_Reserve', 'N778UA_Reserve'
        ],
        'Type': [
            'B77W', 'A320', 'A320', 'B738', 'B752', 'A319', 'B739', 'B777', 'B739', 'B78X',
            'B738', 'B777', 'A320', 'B739', 'B738', 'B78X', 'A319', 'B77W', 'B752', 'B739',
            'B739', 'A320', 'B777'
        ],
        'Location': [
            'ORD', 'EWR', 'EWR', 'IAH', 'LAX', 'EWR', 'DEN', 'EWR', 'ORD', 'SFO',
            'DEN', 'SFO', 'LAX', 'ORD', 'IAH', 'EWR', 'SFO', 'LAX', 'ORD', 'IAD',
            'ORD', 'EWR', 'SFO' # Reserve aircraft at key hubs
        ]
    }
    aircraft_df = pd.DataFrame(aircraft_data).set_index('TailNum')
    
    return flights_df, aircraft_df

def style_schedule(df):
    """Applies color styling to the schedule based on status."""
    def highlight_status(row):
        style = [''] * len(row)
        # Colors from the uploaded image
        if row['Status'] == 'Delayed':
            style = ['background-color: #FFF3CD; color: #664D03'] * len(row)
        elif row['Status'] == 'Canceled':
            style = ['background-color: #F8D7DA; color: #58151A'] * len(row)
        elif row['Status'] == 'Aircraft Swap':
            style = ['background-color: #D1E7DD; color: #0F5132'] * len(row)
        return style
    
    return df.style.apply(highlight_status, axis=1).format({
        'Departure_Orig': '{:%H:%M}', 'Arrival_Orig': '{:%H:%M}',
        'Departure_New': '{:%H:%M}', 'Arrival_New': '{:%H:%M}'
    })

def run_milp_optimization(flights_df, aircraft_df, scenario, objective):
    """
    Builds and solves the MILP model using PuLP.
    Returns the optimized schedule, total cost, and solver log.
    """
    log = f"INFO: Initializing PuLP MILP solver for scenario: '{scenario}'\n"
    log += f"INFO: Setting objective function to: '{objective}'\n"

    model = pulp.LpProblem("Disruption_Recovery_Model", pulp.LpMinimize)

    # --- Cost Parameters ---
    cost_cancel_flight = 50000
    cost_delay_minute = 100
    cost_affected_pax = 200
    cost_aircraft_swap = 15000
    
    if objective == 'Minimize High-Value Passenger Impact':
        cost_affected_hv_pax = 2500
        log += "INFO: Applying high cost penalty for affecting High-Value passengers.\n"
    else:
        cost_affected_hv_pax = 500
        log += "INFO: Applying standard cost penalty for affecting High-Value passengers.\n"

    # --- Decision Variables ---
    flights = flights_df.copy()
    aircraft = aircraft_df.copy()
    cancel_vars = pulp.LpVariable.dicts("IsCanceled", flights.index, cat='Binary')
    delay_vars = pulp.LpVariable.dicts("Delay", flights.index, lowBound=0, upBound=360, cat='Continuous')
    assignment_vars = pulp.LpVariable.dicts("Assign",
                                           [(f, t) for f in flights.index for t in aircraft.index if flights.loc[f, 'AircraftType'] == aircraft.loc[t, 'Type']],
                                           cat='Binary')

    # --- Objective Function ---
    cancellation_cost = pulp.lpSum(
        cancel_vars[f] * (
            cost_cancel_flight +
            flights.loc[f, 'Passengers'] * cost_affected_pax +
            flights.loc[f, 'HighValuePax'] * cost_affected_hv_pax
        ) for f in flights.index
    )
    delay_cost = pulp.lpSum(delay_vars[f] * cost_delay_minute for f in flights.index)
    swap_cost = pulp.lpSum(
        assignment_vars.get((f, t), 0) * cost_aircraft_swap
        for f, t in assignment_vars.keys() if 'Reserve' in t
    )
    model += cancellation_cost + delay_cost + swap_cost, "Total_Disruption_Cost"

    # --- Constraints ---
    for f in flights.index:
        model += pulp.lpSum(assignment_vars.get((f, t), 0) for t in aircraft.index if (f,t) in assignment_vars) == (1 - cancel_vars[f]), f"Flight_Coverage_{f}"

    for t in aircraft.index:
        model += pulp.lpSum(assignment_vars.get((f, t), 0) for f in flights.index if (f,t) in assignment_vars) <= 1, f"Aircraft_Utilization_{t}"

    for f, t in assignment_vars.keys():
        if aircraft.loc[t, 'Location'] != flights.loc[f, 'Origin']:
            model += assignment_vars[(f, t)] == 0, f"Aircraft_Location_{f}_{t}"

    # --- Scenario-Specific Constraints ---
    if scenario == "Thunderstorms at EWR":
        ewr_flights = flights[flights['Origin'] == 'EWR'].index
        on_time_ish_vars = pulp.LpVariable.dicts("IsOnTimeIsh", ewr_flights, cat='Binary')
        for f in ewr_flights:
            model += delay_vars[f] <= 89.9 + (1 - on_time_ish_vars[f]) * 1000
            model += delay_vars[f] >= 90 * (1 - on_time_ish_vars[f])
        model += pulp.lpSum(on_time_ish_vars[f] for f in ewr_flights) <= 2, "EWR_Capacity_Constraint"
        log += "CONSTRAINT: EWR capacity reduced. Max 2 departures with less than 90 min delay.\n"

    elif scenario == "Mechanical Issue at ORD":
        grounded_aircraft = 'N730UA'
        log += f"CONSTRAINT: Aircraft {grounded_aircraft} is grounded at ORD for unscheduled maintenance.\n"
        model += pulp.lpSum(assignment_vars.get((f, grounded_aircraft), 0) for f in flights.index if (f, grounded_aircraft) in assignment_vars) == 0, f"Ground_Aircraft_{grounded_aircraft}"

    # --- Solve and Parse ---
    log += "INFO: Solving optimization problem...\n"
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    log += f"INFO: Solver status: {pulp.LpStatus[model.status]}\n"

    recovered_schedule = flights_df.copy()
    recovered_schedule['Status'] = 'On Time'
    recovered_schedule['Departure_New'] = recovered_schedule['Departure_Orig']
    recovered_schedule['Arrival_New'] = recovered_schedule['Arrival_Orig']
    recovered_schedule['Assigned_Aircraft'] = ''
    recovered_schedule['Delay_Minutes'] = 0

    for f_idx in flights.index:
        if cancel_vars[f_idx].varValue > 0.9:
            recovered_schedule.loc[f_idx, 'Status'] = 'Canceled'
            recovered_schedule.loc[f_idx, 'Departure_New'] = pd.NaT
            recovered_schedule.loc[f_idx, 'Arrival_New'] = pd.NaT
            log += f"ACTION: Canceling flight {f_idx}.\n"
        else:
            delay_minutes = int(delay_vars[f_idx].varValue)
            recovered_schedule.loc[f_idx, 'Delay_Minutes'] = delay_minutes
            if delay_minutes > 0:
                recovered_schedule.loc[f_idx, 'Status'] = 'Delayed'
                recovered_schedule.loc[f_idx, 'Departure_New'] += pd.Timedelta(minutes=delay_minutes)
                recovered_schedule.loc[f_idx, 'Arrival_New'] += pd.Timedelta(minutes=delay_minutes)
                log += f"ACTION: Delaying flight {f_idx} by {delay_minutes} minutes.\n"
            
            assigned_tail = next((t_idx for t_idx in aircraft.index if (f_idx, t_idx) in assignment_vars and assignment_vars[(f_idx, t_idx)].varValue > 0.9), 'N/A')
            recovered_schedule.loc[f_idx, 'Assigned_Aircraft'] = assigned_tail
            
            original_aircrafts = aircraft_df[(aircraft_df['Location'] == recovered_schedule.loc[f_idx, 'Origin']) & (aircraft_df['Type'] == recovered_schedule.loc[f_idx, 'AircraftType'])]
            if not original_aircrafts.empty:
                original_aircraft = original_aircrafts.index[0]
                if assigned_tail != original_aircraft and recovered_schedule.loc[f_idx, 'Status'] != 'Canceled':
                     recovered_schedule.loc[f_idx, 'Status'] = 'Aircraft Swap'
                     log += f"ACTION: Swapping aircraft for flight {f_idx} to {assigned_tail}.\n"

    total_cost = pulp.value(model.objective)
    log += f"INFO: Solver finished. Optimal solution cost: ${total_cost:,.2f}\n"
    return recovered_schedule, total_cost, log

# --- UI Layout ---

st.title("✈️ United Airlines - Proactive Disruption Management")
st.markdown("A dashboard to visualize and manage operational disruptions using a live **Mixed-Integer Linear Programming (MILP)** model.")

if 'original_flights' not in st.session_state:
    flights, aircraft = get_initial_data()
    st.session_state.original_flights = flights
    st.session_state.original_aircraft = aircraft

with st.sidebar:
    st.header("Scenario Simulation")
    disruption_scenario = st.selectbox("Select Disruption Scenario:", ("Thunderstorms at EWR", "Mechanical Issue at ORD"), key='scenario', help="Choose a pre-defined disruption to simulate the model's response.")
    optimization_objective = st.selectbox("Select Optimization Goal:", ("Minimize High-Value Passenger Impact", "Minimize Total Cost"), key='objective', help="This changes the objective function of the underlying MILP model.")
    st.markdown("---")
    run_button = st.button("Run Recovery Optimization", type="primary", use_container_width=True)
    st.markdown("---")
    st.info("This UI runs a real PuLP solver to find a mathematically optimal recovery plan based on your selections.")

if not run_button:
    st.subheader("Today's Original Flight Schedule")
    display_cols = ['Origin', 'Destination', 'AircraftType', 'Departure_Orig', 'Arrival_Orig', 'Passengers', 'HighValuePax']
    st.dataframe(st.session_state.original_flights[display_cols], use_container_width=True)
    st.warning("⬅️ Please select a scenario and click 'Run Recovery Optimization' in the sidebar to begin.")
else:
    with st.spinner('Running MILP solver... This may take a moment.'):
        recovered_df, cost, log_output = run_milp_optimization(st.session_state.original_flights, st.session_state.original_aircraft, disruption_scenario, optimization_objective)
        st.session_state.recovered_df = recovered_df
        st.session_state.cost = cost
        st.session_state.log_output = log_output

    st.subheader(f"Optimized Recovery Plan: {st.session_state.scenario}")

    orig_df = st.session_state.original_flights
    rec_df = st.session_state.recovered_df
    canceled_flights = rec_df[rec_df['Status'] == 'Canceled']
    delayed_flights = rec_df[rec_df['Status'] == 'Delayed']
    swapped_flights = rec_df[rec_df['Status'] == 'Aircraft Swap']
    orig_pax = orig_df['Passengers'].sum()
    pax_on_canceled = canceled_flights['Passengers'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Canceled Flights", f"{len(canceled_flights)}")
    col2.metric("Delayed Flights", f"{len(delayed_flights)}")
    col3.metric("Affected Passengers", f"{pax_on_canceled}", delta=f"-{round((pax_on_canceled/orig_pax)*100, 1) if orig_pax > 0 else 0}% of total", delta_color="inverse")
    col4.metric("Optimal Plan Cost", f"${st.session_state.cost:,.0f}")
    
    st.markdown("---")
    st.subheader("Recovery Schedule Comparison")
    display_cols = ['Origin', 'Destination', 'Status', 'Departure_Orig', 'Departure_New', 'Arrival_New', 'Assigned_Aircraft', 'Passengers', 'HighValuePax']
    styled_df = style_schedule(rec_df[display_cols].rename(columns={'HighValuePax': 'HV Pax'}))
    st.dataframe(styled_df, use_container_width=True, height=500) # Increased height for more data

    with st.expander("View Cost & Passenger Impact Analysis", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cost Breakdown")
            # Calculate cost components for the chart
            cost_delay_minute = 100
            cost_cancel_flight = 50000
            cost_affected_pax = 200
            cost_aircraft_swap = 15000
            cost_affected_hv_pax = 2500 if st.session_state.objective == 'Minimize High-Value Passenger Impact' else 500

            delay_cost_total = rec_df['Delay_Minutes'].sum() * cost_delay_minute
            swap_cost_total = len(swapped_flights) * cost_aircraft_swap
            cancellation_cost_total = (len(canceled_flights) * cost_cancel_flight) + \
                                      (canceled_flights['Passengers'].sum() * cost_affected_pax) + \
                                      (canceled_flights['HighValuePax'].sum() * cost_affected_hv_pax)

            cost_data = pd.DataFrame({
                'Cost Type': ['Delay Costs', 'Cancellation & Pax Costs', 'Aircraft Swap Costs'],
                'Cost (USD)': [delay_cost_total, cancellation_cost_total, swap_cost_total]
            })
            st.bar_chart(cost_data.set_index('Cost Type'))

        with col2:
            st.subheader("High-Value Passenger Impact")
            orig_hv_pax_total = orig_df['HighValuePax'].sum()
            hv_pax_on_canceled = canceled_flights['HighValuePax'].sum()
            st.metric("Global Services / 1K Members on Canceled Flights", f"{hv_pax_on_canceled}")
            st.markdown(f"""
            The optimization model strategically decides which flights to delay or cancel. When the objective is **'{st.session_state.objective}'**, the model's decisions are heavily influenced by the number of high-value passengers on each flight.
            - **Total High-Value Pax in Schedule:** {orig_hv_pax_total}
            - **HV Pax on Canceled Flights:** {hv_pax_on_canceled}
            """)

    with st.expander("View MILP Solver Log"):
        st.text_area("Solver Log", value=st.session_state.log_output, height=300, disabled=True)