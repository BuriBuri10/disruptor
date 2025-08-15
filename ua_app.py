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
            'UA1118', 'UA123', 'UA789', 'UA2001', 'UA555', 'UA300', 'UA1812', 'UA950', 'UA401', 'UA1602',
            # --- New Indian Routes ---
            'UA82', 'UA104', 'UA9421', 'UA9422'
        ],
        'Origin': [
            'ORD', 'EWR', 'EWR', 'IAH', 'LAX', 'EWR', 'DEN', 'EWR', 'ORD', 'SFO',
            'DEN', 'SFO', 'LAX', 'ORD', 'IAH', 'EWR', 'SFO', 'LAX', 'ORD', 'IAD',
            # --- New Indian Routes ---
            'EWR', 'SFO', 'DEL', 'BOM'
        ],
        'Destination': [
            'EWR', 'MIA', 'ORD', 'DEN', 'ORD', 'FLL', 'IAH', 'SFO', 'LAX', 'EWR',
            'LAX', 'IAD', 'JFK', 'BOS', 'MCO', 'SFO', 'DEN', 'EWR', 'MIA', 'LAX',
            # --- New Indian Routes ---
            'DEL', 'BLR', 'BOM', 'DEL'
        ],
        'AircraftType': [
            'B77W', 'A320', 'A320', 'B738', 'B752', 'A319', 'B739', 'B777', 'B739', 'B78X',
            'B738', 'B777', 'A320', 'B739', 'B738', 'B78X', 'A319', 'B77W', 'B752', 'B739',
            # --- New Indian Routes ---
            'B78X', 'B78X', 'A320', 'A320'
        ],
        'Departure_Orig': pd.to_datetime([
            '2025-08-15 07:00', '2025-08-15 07:15', '2025-08-15 08:15', '2025-08-15 08:30', '2025-08-15 08:45',
            '2025-08-15 09:00', '2025-08-15 09:20', '2025-08-15 09:45', '2025-08-15 10:00', '2025-08-15 10:30',
            '2025-08-15 11:00', '2025-08-15 11:25', '2025-08-15 12:00', '2025-08-15 12:15', '2025-08-15 12:45',
            '2025-08-15 13:00', '2025-08-15 13:30', '2025-08-15 14:00', '2025-08-15 14:20', '2025-08-15 15:00',
            # --- New Indian Routes (Times in UTC for consistency) ---
            '2025-08-15 20:00', '2025-08-15 21:30', '2025-08-15 14:00', '2025-08-15 17:00'
        ]),
        'Arrival_Orig': pd.to_datetime([
            '2025-08-15 10:30', '2025-08-15 10:15', '2025-08-15 10:00', '2025-08-15 10:00', '2025-08-15 14:15',
            '2025-08-15 12:00', '2025-08-15 12:30', '2025-08-15 12:45', '2025-08-15 13:00', '2025-08-15 18:30',
            '2025-08-15 13:00', '2025-08-15 19:30', '2025-08-15 20:15', '2025-08-15 14:30', '2025-08-15 15:15',
            '2025-08-15 16:00', '2025-08-15 15:30', '2025-08-15 22:00', '2025-08-15 17:00', '2025-08-15 17:30',
            # --- New Indian Routes (Times in UTC for consistency) ---
            '2025-08-16 20:30', '2025-08-16 23:00', '2025-08-15 16:15', '2025-08-15 19:15'
        ]),
        'Passengers': [
            350, 150, 150, 166, 176, 128, 179, 250, 179, 300,
            166, 250, 150, 179, 166, 300, 128, 350, 176, 179,
            # --- New Indian Routes ---
            280, 290, 145, 148
        ],
        'HighValuePax': [
            40, 10, 9, 12, 18, 8, 14, 20, 15, 35,
            11, 22, 13, 19, 15, 33, 7, 45, 16, 21,
            # --- New Indian Routes ---
            45, 50, 15, 12
        ],
    }
    flights_df = pd.DataFrame(flights_data).set_index('FlightID')

    aircraft_data = {
        'TailNum': [
            'N777UA', 'N320UA', 'N321UA', 'N738UA', 'N752UA', 'N319UA', 'N731UA', 'N770UA', 'N730UA', 'N787UA',
            'N739UA', 'N771UA', 'N322UA', 'N732UA', 'N733UA', 'N788UA', 'N318UA', 'N772UA', 'N753UA', 'N734UA',
            'N735UA_Reserve', 'N325UA_Reserve', 'N778UA_Reserve',
            # --- New Aircraft for Indian Routes ---
            'N789UA', 'N780UA', 'N323UA', 'N324UA'
        ],
        'Type': [
            'B77W', 'A320', 'A320', 'B738', 'B752', 'A319', 'B739', 'B777', 'B739', 'B78X',
            'B738', 'B777', 'A320', 'B739', 'B738', 'B78X', 'A319', 'B77W', 'B752', 'B739',
            'B739', 'A320', 'B777',
            # --- New Aircraft for Indian Routes ---
            'B78X', 'B78X', 'A320', 'A320'
        ],
        'Location': [
            'ORD', 'EWR', 'EWR', 'IAH', 'LAX', 'EWR', 'DEN', 'EWR', 'ORD', 'SFO',
            'DEN', 'SFO', 'LAX', 'ORD', 'IAH', 'EWR', 'SFO', 'LAX', 'ORD', 'IAD',
            'ORD', 'EWR', 'SFO', # Reserve aircraft at key hubs
            # --- New Aircraft for Indian Routes ---
            'EWR', 'SFO', 'DEL', 'BOM'
        ]
    }
    aircraft_df = pd.DataFrame(aircraft_data).set_index('TailNum')
    
    return flights_df, aircraft_df

def style_schedule(df):
    """Applies color styling to the schedule based on status."""
    def highlight_status(row):
        style = [''] * len(row)
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
    }, na_rep='-')

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
    
    # --- New Indian Scenarios ---
    elif scenario == "Monsoon Rains at BOM":
        bom_flights = flights[flights['Origin'] == 'BOM'].index
        on_time_departure_vars = pulp.LpVariable.dicts("IsOnTimeBOM", bom_flights, cat='Binary')
        for f in bom_flights:
            # If delay is > 30 mins, IsOnTimeBOM is 0
            model += delay_vars[f] <= 29.9 + (1 - on_time_departure_vars[f]) * 1000
        # Only one flight can depart "on time" (with less than 30 min delay)
        model += pulp.lpSum(on_time_departure_vars[f] for f in bom_flights) <= 0, "BOM_Monsoon_Capacity"
        log += "CONSTRAINT: BOM capacity severely reduced due to monsoon. No departures with less than 30 min delay allowed.\n"

    elif scenario == "Dense Fog at DEL":
        # Simulate an aircraft not being CAT III compliant for low visibility landings
        grounded_aircraft = 'N323UA' 
        log += f"CONSTRAINT: Aircraft {grounded_aircraft} at DEL is not CAT III compliant and is grounded due to dense fog.\n"
        model += pulp.lpSum(assignment_vars.get((f, grounded_aircraft), 0) for f in flights.index if (f, grounded_aircraft) in assignment_vars) == 0, f"Ground_Aircraft_Fog_{grounded_aircraft}"


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
                # Find the first non-reserve aircraft to be considered the "original"
                original_aircraft = next((ac for ac in original_aircrafts.index if 'Reserve' not in ac), original_aircrafts.index[0])
                if assigned_tail != original_aircraft and recovered_schedule.loc[f_idx, 'Status'] != 'Canceled':
                        recovered_schedule.loc[f_idx, 'Status'] = 'Aircraft Swap'
                        log += f"ACTION: Swapping aircraft for flight {f_idx} to {assigned_tail}.\n"

    total_cost = pulp.value(model.objective)
    log += f"INFO: Solver finished. Optimal solution cost: ${total_cost:,.2f}\n"
    return recovered_schedule, total_cost, log

# --- UI Layout ---

st.title("United Airlines Proactive Disruption Management ✈")
st.markdown("A dashboard to visualize and manage operational disruptions using a live **Mixed-Integer Linear Programming (MILP)** model.")

if 'original_flights' not in st.session_state:
    flights, aircraft = get_initial_data()
    st.session_state.original_flights = flights
    st.session_state.original_aircraft = aircraft

with st.sidebar:
    st.header("Scenario Simulation")
    # --- Added New Indian Scenarios to the list ---
    disruption_scenario = st.selectbox(
        "Select Disruption Scenario:", 
        ("Thunderstorms at EWR", "Mechanical Issue at ORD", "Monsoon Rains at BOM", "Dense Fog at DEL"), 
        key='scenario', 
        help="Choose a pre-defined disruption to simulate the model's response."
    )
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
    st.dataframe(styled_df, use_container_width=True, height=700)

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






# import streamlit as st
# import pandas as pd
# import numpy as np
# import pulp
# import time

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="United Airlines Disruption Management",
#     page_icon="✈️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Helper Functions & Data Setup ---

# @st.cache_data
# def get_initial_data():
#     """
#     Creates the initial flight and aircraft schedules with more realistic and extensive data.
#     This data serves as the input for the MILP model.
#     """
#     flights_data = {
#         'FlightID': [
#             'UA1989', 'UA2428', 'UA635', 'UA1722', 'UA511', 'UA2402', 'UA331', 'UA482', 'UA1992', 'UA211',
#             'UA1118', 'UA123', 'UA789', 'UA2001', 'UA555', 'UA300', 'UA1812', 'UA950', 'UA401', 'UA1602',
#             # --- New Indian Routes ---
#             'UA82', 'UA104', 'UA9421', 'UA9422'
#         ],
#         'Origin': [
#             'ORD', 'EWR', 'EWR', 'IAH', 'LAX', 'EWR', 'DEN', 'EWR', 'ORD', 'SFO',
#             'DEN', 'SFO', 'LAX', 'ORD', 'IAH', 'EWR', 'SFO', 'LAX', 'ORD', 'IAD',
#             # --- New Indian Routes ---
#             'EWR', 'SFO', 'DEL', 'BOM'
#         ],
#         'Destination': [
#             'EWR', 'MIA', 'ORD', 'DEN', 'ORD', 'FLL', 'IAH', 'SFO', 'LAX', 'EWR',
#             'LAX', 'IAD', 'JFK', 'BOS', 'MCO', 'SFO', 'DEN', 'EWR', 'MIA', 'LAX',
#             # --- New Indian Routes ---
#             'DEL', 'BLR', 'BOM', 'DEL'
#         ],
#         'AircraftType': [
#             'B77W', 'A320', 'A320', 'B738', 'B752', 'A319', 'B739', 'B777', 'B739', 'B78X',
#             'B738', 'B777', 'A320', 'B739', 'B738', 'B78X', 'A319', 'B77W', 'B752', 'B739',
#             # --- New Indian Routes ---
#             'B78X', 'B78X', 'A320', 'A320'
#         ],
#         'Departure_Orig': pd.to_datetime([
#             '2025-08-15 07:00', '2025-08-15 07:15', '2025-08-15 08:15', '2025-08-15 08:30', '2025-08-15 08:45',
#             '2025-08-15 09:00', '2025-08-15 09:20', '2025-08-15 09:45', '2025-08-15 10:00', '2025-08-15 10:30',
#             '2025-08-15 11:00', '2025-08-15 11:25', '2025-08-15 12:00', '2025-08-15 12:15', '2025-08-15 12:45',
#             '2025-08-15 13:00', '2025-08-15 13:30', '2025-08-15 14:00', '2025-08-15 14:20', '2025-08-15 15:00',
#             # --- New Indian Routes (Times in UTC for consistency) ---
#             '2025-08-15 20:00', '2025-08-15 21:30', '2025-08-15 14:00', '2025-08-15 17:00'
#         ]),
#         'Arrival_Orig': pd.to_datetime([
#             '2025-08-15 10:30', '2025-08-15 10:15', '2025-08-15 10:00', '2025-08-15 10:00', '2025-08-15 14:15',
#             '2025-08-15 12:00', '2025-08-15 12:30', '2025-08-15 12:45', '2025-08-15 13:00', '2025-08-15 18:30',
#             '2025-08-15 13:00', '2025-08-15 19:30', '2025-08-15 20:15', '2025-08-15 14:30', '2025-08-15 15:15',
#             '2025-08-15 16:00', '2025-08-15 15:30', '2025-08-15 22:00', '2025-08-15 17:00', '2025-08-15 17:30',
#             # --- New Indian Routes (Times in UTC for consistency) ---
#             '2025-08-16 20:30', '2025-08-16 23:00', '2025-08-15 16:15', '2025-08-15 19:15'
#         ]),
#         'Passengers': [
#             350, 150, 150, 166, 176, 128, 179, 250, 179, 300,
#             166, 250, 150, 179, 166, 300, 128, 350, 176, 179,
#             # --- New Indian Routes ---
#             280, 290, 145, 148
#         ],
#         'HighValuePax': [
#             40, 10, 9, 12, 18, 8, 14, 20, 15, 35,
#             11, 22, 13, 19, 15, 33, 7, 45, 16, 21,
#             # --- New Indian Routes ---
#             45, 50, 15, 12
#         ],
#     }
#     flights_df = pd.DataFrame(flights_data).set_index('FlightID')

#     aircraft_data = {
#         'TailNum': [
#             'N777UA', 'N320UA', 'N321UA', 'N738UA', 'N752UA', 'N319UA', 'N731UA', 'N770UA', 'N730UA', 'N787UA',
#             'N739UA', 'N771UA', 'N322UA', 'N732UA', 'N733UA', 'N788UA', 'N318UA', 'N772UA', 'N753UA', 'N734UA',
#             'N735UA_Reserve', 'N325UA_Reserve', 'N778UA_Reserve',
#             # --- New Aircraft for Indian Routes ---
#             'N789UA', 'N780UA', 'N323UA', 'N324UA'
#         ],
#         'Type': [
#             'B77W', 'A320', 'A320', 'B738', 'B752', 'A319', 'B739', 'B777', 'B739', 'B78X',
#             'B738', 'B777', 'A320', 'B739', 'B738', 'B78X', 'A319', 'B77W', 'B752', 'B739',
#             'B739', 'A320', 'B777',
#             # --- New Aircraft for Indian Routes ---
#             'B78X', 'B78X', 'A320', 'A320'
#         ],
#         'Location': [
#             'ORD', 'EWR', 'EWR', 'IAH', 'LAX', 'EWR', 'DEN', 'EWR', 'ORD', 'SFO',
#             'DEN', 'SFO', 'LAX', 'ORD', 'IAH', 'EWR', 'SFO', 'LAX', 'ORD', 'IAD',
#             'ORD', 'EWR', 'SFO', # Reserve aircraft at key hubs
#             # --- New Aircraft for Indian Routes ---
#             'EWR', 'SFO', 'DEL', 'BOM'
#         ]
#     }
#     aircraft_df = pd.DataFrame(aircraft_data).set_index('TailNum')
    
#     return flights_df, aircraft_df

# def style_schedule(df):
#     """Applies color styling to the schedule based on status."""
#     def highlight_status(row):
#         style = [''] * len(row)
#         if row['Status'] == 'Delayed':
#             style = ['background-color: #FFF3CD; color: #664D03'] * len(row)
#         elif row['Status'] == 'Canceled':
#             style = ['background-color: #F8D7DA; color: #58151A'] * len(row)
#         elif row['Status'] == 'Aircraft Swap':
#             style = ['background-color: #D1E7DD; color: #0F5132'] * len(row)
#         return style
    
#     return df.style.apply(highlight_status, axis=1).format({
#         'Departure_Orig': '{:%H:%M}', 'Arrival_Orig': '{:%H:%M}',
#         'Departure_New': '{:%H:%M}', 'Arrival_New': '{:%H:%M}'
#     })

# def run_milp_optimization(flights_df, aircraft_df, scenario, objective):
#     """
#     Builds and solves the MILP model using PuLP.
#     Returns the optimized schedule, total cost, and solver log.
#     """
#     log = f"INFO: Initializing PuLP MILP solver for scenario: '{scenario}'\n"
#     log += f"INFO: Setting objective function to: '{objective}'\n"

#     model = pulp.LpProblem("Disruption_Recovery_Model", pulp.LpMinimize)

#     # --- Cost Parameters ---
#     cost_cancel_flight = 50000
#     cost_delay_minute = 100
#     cost_affected_pax = 200
#     cost_aircraft_swap = 15000
    
#     if objective == 'Minimize High-Value Passenger Impact':
#         cost_affected_hv_pax = 2500
#         log += "INFO: Applying high cost penalty for affecting High-Value passengers.\n"
#     else:
#         cost_affected_hv_pax = 500
#         log += "INFO: Applying standard cost penalty for affecting High-Value passengers.\n"

#     # --- Decision Variables ---
#     flights = flights_df.copy()
#     aircraft = aircraft_df.copy()
#     cancel_vars = pulp.LpVariable.dicts("IsCanceled", flights.index, cat='Binary')
#     delay_vars = pulp.LpVariable.dicts("Delay", flights.index, lowBound=0, upBound=360, cat='Continuous')
#     assignment_vars = pulp.LpVariable.dicts("Assign",
#                                               [(f, t) for f in flights.index for t in aircraft.index if flights.loc[f, 'AircraftType'] == aircraft.loc[t, 'Type']],
#                                               cat='Binary')

#     # --- Objective Function ---
#     cancellation_cost = pulp.lpSum(
#         cancel_vars[f] * (
#             cost_cancel_flight +
#             flights.loc[f, 'Passengers'] * cost_affected_pax +
#             flights.loc[f, 'HighValuePax'] * cost_affected_hv_pax
#         ) for f in flights.index
#     )
#     delay_cost = pulp.lpSum(delay_vars[f] * cost_delay_minute for f in flights.index)
#     swap_cost = pulp.lpSum(
#         assignment_vars.get((f, t), 0) * cost_aircraft_swap
#         for f, t in assignment_vars.keys() if 'Reserve' in t
#     )
#     model += cancellation_cost + delay_cost + swap_cost, "Total_Disruption_Cost"

#     # --- Constraints ---
#     for f in flights.index:
#         model += pulp.lpSum(assignment_vars.get((f, t), 0) for t in aircraft.index if (f,t) in assignment_vars) == (1 - cancel_vars[f]), f"Flight_Coverage_{f}"

#     for t in aircraft.index:
#         model += pulp.lpSum(assignment_vars.get((f, t), 0) for f in flights.index if (f,t) in assignment_vars) <= 1, f"Aircraft_Utilization_{t}"

#     for f, t in assignment_vars.keys():
#         if aircraft.loc[t, 'Location'] != flights.loc[f, 'Origin']:
#             model += assignment_vars[(f, t)] == 0, f"Aircraft_Location_{f}_{t}"

#     # --- Scenario-Specific Constraints ---
#     if scenario == "Thunderstorms at EWR":
#         ewr_flights = flights[flights['Origin'] == 'EWR'].index
#         on_time_ish_vars = pulp.LpVariable.dicts("IsOnTimeIsh", ewr_flights, cat='Binary')
#         for f in ewr_flights:
#             model += delay_vars[f] <= 89.9 + (1 - on_time_ish_vars[f]) * 1000
#             model += delay_vars[f] >= 90 * (1 - on_time_ish_vars[f])
#         model += pulp.lpSum(on_time_ish_vars[f] for f in ewr_flights) <= 2, "EWR_Capacity_Constraint"
#         log += "CONSTRAINT: EWR capacity reduced. Max 2 departures with less than 90 min delay.\n"

#     elif scenario == "Mechanical Issue at ORD":
#         grounded_aircraft = 'N730UA'
#         log += f"CONSTRAINT: Aircraft {grounded_aircraft} is grounded at ORD for unscheduled maintenance.\n"
#         model += pulp.lpSum(assignment_vars.get((f, grounded_aircraft), 0) for f in flights.index if (f, grounded_aircraft) in assignment_vars) == 0, f"Ground_Aircraft_{grounded_aircraft}"

#     # --- Solve and Parse ---
#     log += "INFO: Solving optimization problem...\n"
#     model.solve(pulp.PULP_CBC_CMD(msg=0))
#     log += f"INFO: Solver status: {pulp.LpStatus[model.status]}\n"

#     recovered_schedule = flights_df.copy()
#     recovered_schedule['Status'] = 'On Time'
#     recovered_schedule['Departure_New'] = recovered_schedule['Departure_Orig']
#     recovered_schedule['Arrival_New'] = recovered_schedule['Arrival_Orig']
#     recovered_schedule['Assigned_Aircraft'] = ''
#     recovered_schedule['Delay_Minutes'] = 0

#     for f_idx in flights.index:
#         if cancel_vars[f_idx].varValue > 0.9:
#             recovered_schedule.loc[f_idx, 'Status'] = 'Canceled'
#             recovered_schedule.loc[f_idx, 'Departure_New'] = pd.NaT
#             recovered_schedule.loc[f_idx, 'Arrival_New'] = pd.NaT
#             log += f"ACTION: Canceling flight {f_idx}.\n"
#         else:
#             delay_minutes = int(delay_vars[f_idx].varValue)
#             recovered_schedule.loc[f_idx, 'Delay_Minutes'] = delay_minutes
#             if delay_minutes > 0:
#                 recovered_schedule.loc[f_idx, 'Status'] = 'Delayed'
#                 recovered_schedule.loc[f_idx, 'Departure_New'] += pd.Timedelta(minutes=delay_minutes)
#                 recovered_schedule.loc[f_idx, 'Arrival_New'] += pd.Timedelta(minutes=delay_minutes)
#                 log += f"ACTION: Delaying flight {f_idx} by {delay_minutes} minutes.\n"
            
#             assigned_tail = next((t_idx for t_idx in aircraft.index if (f_idx, t_idx) in assignment_vars and assignment_vars[(f_idx, t_idx)].varValue > 0.9), 'N/A')
#             recovered_schedule.loc[f_idx, 'Assigned_Aircraft'] = assigned_tail
            
#             original_aircrafts = aircraft_df[(aircraft_df['Location'] == recovered_schedule.loc[f_idx, 'Origin']) & (aircraft_df['Type'] == recovered_schedule.loc[f_idx, 'AircraftType'])]
#             if not original_aircrafts.empty:
#                 original_aircraft = original_aircrafts.index[0]
#                 if assigned_tail != original_aircraft and recovered_schedule.loc[f_idx, 'Status'] != 'Canceled':
#                         recovered_schedule.loc[f_idx, 'Status'] = 'Aircraft Swap'
#                         log += f"ACTION: Swapping aircraft for flight {f_idx} to {assigned_tail}.\n"

#     total_cost = pulp.value(model.objective)
#     log += f"INFO: Solver finished. Optimal solution cost: ${total_cost:,.2f}\n"
#     return recovered_schedule, total_cost, log

# # --- UI Layout ---

# st.title("✈️ United Airlines - Proactive Disruption Management")
# st.markdown("A dashboard to visualize and manage operational disruptions using a live **Mixed-Integer Linear Programming (MILP)** model.")

# if 'original_flights' not in st.session_state:
#     flights, aircraft = get_initial_data()
#     st.session_state.original_flights = flights
#     st.session_state.original_aircraft = aircraft

# with st.sidebar:
#     st.header("Scenario Simulation")
#     disruption_scenario = st.selectbox("Select Disruption Scenario:", ("Thunderstorms at EWR", "Mechanical Issue at ORD"), key='scenario', help="Choose a pre-defined disruption to simulate the model's response.")
#     optimization_objective = st.selectbox("Select Optimization Goal:", ("Minimize High-Value Passenger Impact", "Minimize Total Cost"), key='objective', help="This changes the objective function of the underlying MILP model.")
#     st.markdown("---")
#     run_button = st.button("Run Recovery Optimization", type="primary", use_container_width=True)
#     st.markdown("---")
#     st.info("This UI runs a real PuLP solver to find a mathematically optimal recovery plan based on your selections.")

# if not run_button:
#     st.subheader("Today's Original Flight Schedule")
#     display_cols = ['Origin', 'Destination', 'AircraftType', 'Departure_Orig', 'Arrival_Orig', 'Passengers', 'HighValuePax']
#     st.dataframe(st.session_state.original_flights[display_cols], use_container_width=True)
#     st.warning("⬅️ Please select a scenario and click 'Run Recovery Optimization' in the sidebar to begin.")
# else:
#     with st.spinner('Running MILP solver... This may take a moment.'):
#         recovered_df, cost, log_output = run_milp_optimization(st.session_state.original_flights, st.session_state.original_aircraft, disruption_scenario, optimization_objective)
#         st.session_state.recovered_df = recovered_df
#         st.session_state.cost = cost
#         st.session_state.log_output = log_output

#     st.subheader(f"Optimized Recovery Plan: {st.session_state.scenario}")

#     orig_df = st.session_state.original_flights
#     rec_df = st.session_state.recovered_df
#     canceled_flights = rec_df[rec_df['Status'] == 'Canceled']
#     delayed_flights = rec_df[rec_df['Status'] == 'Delayed']
#     swapped_flights = rec_df[rec_df['Status'] == 'Aircraft Swap']
#     orig_pax = orig_df['Passengers'].sum()
#     pax_on_canceled = canceled_flights['Passengers'].sum()

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Canceled Flights", f"{len(canceled_flights)}")
#     col2.metric("Delayed Flights", f"{len(delayed_flights)}")
#     col3.metric("Affected Passengers", f"{pax_on_canceled}", delta=f"-{round((pax_on_canceled/orig_pax)*100, 1) if orig_pax > 0 else 0}% of total", delta_color="inverse")
#     col4.metric("Optimal Plan Cost", f"${st.session_state.cost:,.0f}")
    
#     st.markdown("---")
#     st.subheader("Recovery Schedule Comparison")
#     display_cols = ['Origin', 'Destination', 'Status', 'Departure_Orig', 'Departure_New', 'Arrival_New', 'Assigned_Aircraft', 'Passengers', 'HighValuePax']
#     styled_df = style_schedule(rec_df[display_cols].rename(columns={'HighValuePax': 'HV Pax'}))
#     st.dataframe(styled_df, use_container_width=True, height=700) # Increased height for more data

#     with st.expander("View Cost & Passenger Impact Analysis", expanded=True):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Cost Breakdown")
#             # Calculate cost components for the chart
#             cost_delay_minute = 100
#             cost_cancel_flight = 50000
#             cost_affected_pax = 200
#             cost_aircraft_swap = 15000
#             cost_affected_hv_pax = 2500 if st.session_state.objective == 'Minimize High-Value Passenger Impact' else 500

#             delay_cost_total = rec_df['Delay_Minutes'].sum() * cost_delay_minute
#             swap_cost_total = len(swapped_flights) * cost_aircraft_swap
#             cancellation_cost_total = (len(canceled_flights) * cost_cancel_flight) + \
#                                       (canceled_flights['Passengers'].sum() * cost_affected_pax) + \
#                                       (canceled_flights['HighValuePax'].sum() * cost_affected_hv_pax)

#             cost_data = pd.DataFrame({
#                 'Cost Type': ['Delay Costs', 'Cancellation & Pax Costs', 'Aircraft Swap Costs'],
#                 'Cost (USD)': [delay_cost_total, cancellation_cost_total, swap_cost_total]
#             })
#             st.bar_chart(cost_data.set_index('Cost Type'))

#         with col2:
#             st.subheader("High-Value Passenger Impact")
#             orig_hv_pax_total = orig_df['HighValuePax'].sum()
#             hv_pax_on_canceled = canceled_flights['HighValuePax'].sum()
#             st.metric("Global Services / 1K Members on Canceled Flights", f"{hv_pax_on_canceled}")
#             st.markdown(f"""
#             The optimization model strategically decides which flights to delay or cancel. When the objective is **'{st.session_state.objective}'**, the model's decisions are heavily influenced by the number of high-value passengers on each flight.
#             - **Total High-Value Pax in Schedule:** {orig_hv_pax_total}
#             - **HV Pax on Canceled Flights:** {hv_pax_on_canceled}
#             """)

#     with st.expander("View MILP Solver Log"):
#         st.text_area("Solver Log", value=st.session_state.log_output, height=300, disabled=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import time

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="United Airlines Disruption Management",
#     page_icon="✈️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Helper Functions & Mock Data ---

# def get_mock_schedule():
#     """Creates a mock flight schedule as a pandas DataFrame."""
#     data = {
#         'Flight': ['UA482', 'UA1992', 'UA211', 'UA635', 'UA1722', 'UA511', 'UA2402', 'UA331', 'UA1989', 'UA788'],
#         'Origin': ['EWR', 'ORD', 'SFO', 'EWR', 'IAH', 'LAX', 'EWR', 'DEN', 'ORD', 'SFO'],
#         'Destination': ['SFO', 'LAX', 'EWR', 'ORD', 'DEN', 'ORD', 'MIA', 'IAH', 'EWR', 'EWR'],
#         'Aircraft': ['B777', 'B739', 'B78X', 'A320', 'B738', 'B752', 'A319', 'B739', 'B77W', 'A320'],
#         'Departure_Orig': pd.to_datetime(['2025-08-15 08:00', '2025-08-15 08:30', '2025-08-15 09:00', '2025-08-15 09:15', '2025-08-15 09:30', '2025-08-15 10:00', '2025-08-15 10:20', '2025-08-15 10:45', '2025-08-15 11:00', '2025-08-15 11:30']),
#         'Arrival_Orig': pd.to_datetime(['2025-08-15 11:00', '2025-08-15 11:00', '2025-08-15 17:00', '2025-08-15 11:00', '2025-08-15 11:00', '2025-08-15 15:30', '2025-08-15 13:00', '2025-08-15 14:00', '2025-08-15 14:30', '2025-08-15 19:45']),
#         'Passengers': [250, 179, 300, 150, 166, 176, 128, 179, 350, 150],
#         'GS_1K_Pax': [20, 15, 35, 10, 12, 18, 8, 14, 40, 9], # Global Services / 1K
#         'Status': ['On Time'] * 10
#     }
#     df = pd.DataFrame(data)
#     df['Departure_New'] = df['Departure_Orig']
#     df['Arrival_New'] = df['Arrival_Orig']
#     return df

# def style_schedule(df):
#     """Applies color styling to the schedule based on status."""
#     def highlight_status(row):
#         style = [''] * len(row)
#         if row['Status'] == 'Delayed':
#             style = ['background-color: #FFF3CD; color: #664D03'] * len(row)
#         elif row['Status'] == 'Canceled':
#             style = ['background-color: #F8D7DA; color: #58151A'] * len(row)
#         elif row['Status'] == 'Aircraft Swap':
#             style = ['background-color: #D1E7DD; color: #0F5132'] * len(row)
#         return style
#     return df.style.apply(highlight_status, axis=1)


# def run_milp_simulation(schedule_df, scenario, objective):
#     """Simulates the output of the backend MILP solver."""
#     log = f"INFO: Initializing MILP solver for scenario: '{scenario}'\n"
#     log += f"INFO: Setting objective function to: '{objective}'\n"
    
#     recovered_schedule = schedule_df.copy()
    
#     if scenario == "Thunderstorms at EWR":
#         log += "CONSTRAINT: Reduced arrival/departure capacity at EWR.\n"
#         ewr_flights_idx = recovered_schedule[recovered_schedule['Origin'] == 'EWR'].index
        
#         # Delay first EWR flight
#         recovered_schedule.loc[ewr_flights_idx[0], 'Status'] = 'Delayed'
#         recovered_schedule.loc[ewr_flights_idx[0], 'Departure_New'] += pd.Timedelta(hours=2)
#         recovered_schedule.loc[ewr_flights_idx[0], 'Arrival_New'] += pd.Timedelta(hours=2)
#         log += f"ACTION: Delaying flight {recovered_schedule.loc[ewr_flights_idx[0], 'Flight']} by 120 minutes.\n"

#         # Cancel second EWR flight (fewer high-value pax if objective is customer-focused)
#         flight_to_cancel_idx = ewr_flights_idx[2] if objective == 'Minimize High-Value Passenger Impact' else ewr_flights_idx[1]
#         recovered_schedule.loc[flight_to_cancel_idx, 'Status'] = 'Canceled'
#         log += f"ACTION: Canceling flight {recovered_schedule.loc[flight_to_cancel_idx, 'Flight']}.\n"

#         # Swap aircraft for a non-EWR flight to absorb delay
#         recovered_schedule.loc[4, 'Status'] = 'Aircraft Swap'
#         recovered_schedule.loc[4, 'Aircraft'] = 'A321neo (from reserve)'
#         log += f"ACTION: Swapping aircraft for flight {recovered_schedule.loc[4, 'Flight']} to maintain schedule integrity.\n"

#     elif scenario == "Mechanical Issue at ORD":
#         log += "CONSTRAINT: Aircraft B739 (on UA1992) is grounded for unscheduled maintenance.\n"
#         # Delay the flight with the issue
#         recovered_schedule.loc[1, 'Status'] = 'Delayed'
#         recovered_schedule.loc[1, 'Departure_New'] += pd.Timedelta(hours=3, minutes=30)
#         recovered_schedule.loc[1, 'Arrival_New'] += pd.Timedelta(hours=3, minutes=30)
#         log += f"ACTION: Delaying flight UA1992 by 210 minutes.\n"
        
#         # Cancel a different flight to free up an aircraft
#         recovered_schedule.loc[8, 'Status'] = 'Canceled'
#         log += f"ACTION: Canceling flight UA1989 to free up B77W for other routes.\n"


#     log += "INFO: Solver finished. Optimal solution found.\n"
#     return recovered_schedule, log

# # --- UI Layout ---

# st.title("✈️ United Airlines - Proactive Disruption Management")
# st.markdown("A dashboard to visualize and manage operational disruptions using mathematical optimization.")

# # --- Sidebar for Inputs ---
# with st.sidebar:
#     st.header("Scenario Simulation")
    
#     disruption_scenario = st.selectbox(
#         "Select Disruption Scenario:",
#         ("Thunderstorms at EWR", "Mechanical Issue at ORD"),
#         help="Choose a pre-defined disruption to simulate the model's response."
#     )
    
#     optimization_objective = st.selectbox(
#         "Select Optimization Goal:",
#         ("Minimize Total Cost", "Minimize High-Value Passenger Impact"),
#         help="This changes the objective function of the underlying MILP model."
#     )
    
#     st.markdown("---")
    
#     run_button = st.button("Run Recovery Optimization", type="primary")
    
#     st.markdown("---")
#     st.info("This UI simulates the output of a complex Mixed-Integer Linear Programming (MILP) model running on the backend.")


# # --- Main Panel for Outputs ---

# if 'original_schedule' not in st.session_state:
#     st.session_state.original_schedule = get_mock_schedule()

# if not run_button:
#     st.subheader("Today's Original Flight Schedule")
#     st.dataframe(st.session_state.original_schedule, use_container_width=True)
#     st.warning("Please select a scenario and click 'Run Recovery Optimization' in the sidebar to begin.")

# else:
#     with st.spinner('Running MILP solver... This may take a moment.'):
#         recovered_df, log_output = run_milp_simulation(
#             st.session_state.original_schedule,
#             disruption_scenario,
#             optimization_objective
#         )
#         time.sleep(2) # Simulate solver time

#     st.subheader(f"Optimized Recovery Plan: {disruption_scenario}")

#     # --- Key Metrics ---
#     orig_pax = st.session_state.original_schedule['Passengers'].sum()
#     recovered_pax = recovered_df[recovered_df['Status'] != 'Canceled']['Passengers'].sum()
    
#     canceled_flights = len(recovered_df[recovered_df['Status'] == 'Canceled'])
#     delayed_flights = len(recovered_df[recovered_df['Status'] == 'Delayed'])

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Canceled Flights", f"{canceled_flights}", help="Total number of flights canceled in the recovery plan.")
#     col2.metric("Delayed Flights", f"{delayed_flights}", help="Total number of flights delayed in the recovery plan.")
#     col3.metric("Affected Passengers", f"{orig_pax - recovered_pax}", delta=f"-{round(((orig_pax - recovered_pax)/orig_pax)*100, 1)}%", delta_color="inverse", help="Passengers on canceled flights.")
#     col4.metric("Estimated Cost", "$1.2M", delta="$350K vs manual", delta_color="normal", help="Estimated cost of the recovery plan vs. a non-optimized manual recovery.")
    
#     st.markdown("---")

#     # --- Schedules Display ---
#     st.subheader("Recovery Schedule Comparison")
    
#     styled_df = style_schedule(recovered_df)
#     st.dataframe(styled_df, use_container_width=True, height=385)

#     # --- Analysis & Logs ---
#     with st.expander("View Cost & Passenger Impact Analysis"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Cost Breakdown")
#             cost_data = pd.DataFrame({
#                 'Cost Type': ['Passenger Re-accommodation', 'Crew Overtime/Rescheduling', 'EU261 Compensation', 'Lost Revenue'],
#                 'Cost (USD)': [450000, 200000, 150000, 400000]
#             })
#             st.bar_chart(cost_data, x='Cost Type', y='Cost (USD)')

#         with col2:
#             st.subheader("Affected High-Value Passengers")
#             orig_hv_pax = st.session_state.original_schedule['GS_1K_Pax'].sum()
#             recovered_hv_pax = recovered_df[recovered_df['Status'] != 'Canceled']['GS_1K_Pax'].sum()
            
#             st.metric("Global Services / 1K Members Affected", f"{orig_hv_pax - recovered_hv_pax}")
#             st.markdown(f"""
#             The optimization model, especially when focused on minimizing passenger impact, strategically prioritizes flights with higher numbers of top-tier elite members.
#             - **Original HV Pax on Canceled Flights:** {st.session_state.original_schedule.loc[recovered_df['Status'] == 'Canceled', 'GS_1K_Pax'].sum()}
#             - **New Plan HV Pax on Canceled Flights:** {recovered_df.loc[recovered_df['Status'] == 'Canceled', 'GS_1K_Pax'].sum()}
#             """)

#     with st.expander("View MILP Solver Log"):
#         st.text_area("Solver Log", value=log_output, height=250, disabled=True)

