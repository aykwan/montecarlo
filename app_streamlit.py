import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dataclasses import dataclass

# -- [Include all your existing classes (NewsvendorParams, DemandDistribution, NewsvendorSimulation) here] --
@dataclass
class NewsvendorParams:
    """Parameters for the newsvendor model."""
    purchase_price: float = 30  # Cost to purchase one unit (in pence)
    selling_price: float = 75   # Revenue from selling one unit (in pence)
    salvage_value: float = 5    # Value of recycling one unsold unit (in pence)
    batch_size: int = 10        # Units can only be ordered in batches of this size
    planning_days: int = 20     # Number of days in the planning period

class DemandDistribution:
    """Represents demand distributions for different day types."""
    
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        # Define demand distributions for different day types
        # Format: {quantity: probability}
        self.distributions = {
            'good': {30: 0.05, 40: 0.10, 50: 0.15, 60: 0.20, 
                     70: 0.25, 80: 0.15, 90: 0.08, 100: 0.02},
            'fair': {20: 0.10, 30: 0.15, 40: 0.25, 50: 0.30, 
                     60: 0.15, 70: 0.05},
            'poor': {10: 0.15, 20: 0.35, 30: 0.35, 40: 0.15}
        }
        # Define probabilities of day types
        self.day_type_probs = {'good': 0.3, 'fair': 0.5, 'poor': 0.2}
    
    def sample_day_type(self):
        """Sample a random day type based on probabilities."""
        day_types = list(self.day_type_probs.keys())
        probs = list(self.day_type_probs.values())
        return self.rng.choice(day_types, p=probs)
    
    def sample_demand(self, day_type):
        """Sample demand from the distribution for a given day type."""
        dist = self.distributions[day_type]
        quantities = list(dist.keys())
        probs = list(dist.values())
        return self.rng.choice(quantities, p=probs)
		
class NewsvendorSimulation:
    """Monte Carlo simulation for the newsvendor problem."""
    
    def __init__(self, params=None, seed=None):
        self.params = params if params else NewsvendorParams()
        self.demand_dist = DemandDistribution(seed)
        self.seed = seed
    
    def run_single_simulation(self, order_quantity):
        """Run a single simulation of the planning period."""
        results = []
        
        for day in range(1, self.params.planning_days + 1):
            day_type = self.demand_dist.sample_day_type()
            demand = self.demand_dist.sample_demand(day_type)
            
            # Calculate actual sales (limited by available inventory)
            sales = min(order_quantity, demand)
            
            # Calculate revenue components
            sales_revenue = sales * self.params.selling_price
            leftover = max(0, order_quantity - demand)
            salvage_revenue = leftover * self.params.salvage_value
            purchase_cost = order_quantity * self.params.purchase_price
            
            # Calculate profit
            profit = sales_revenue + salvage_revenue - purchase_cost
            
            results.append({
                'day': day,
                'day_type': day_type,
                'demand': demand,
                'sales': sales,
                'leftover': leftover,
                'shortage': max(0, demand - order_quantity),
                'sales_revenue': sales_revenue,
                'salvage_revenue': salvage_revenue,
                'purchase_cost': purchase_cost,
                'profit': profit
            })
        
        return pd.DataFrame(results)
    
    def run_monte_carlo(self, order_quantities, num_simulations=1000):
        """Run Monte Carlo simulation for multiple order quantities."""
        results = {}
        
        for q in order_quantities:
            # Ensure order quantity is a multiple of batch size
            if q % self.params.batch_size != 0:
                continue
                
            total_profits = []
            
            for sim in range(num_simulations):
                # Set a different seed for each simulation
                sim_seed = None if self.seed is None else self.seed + sim
                self.demand_dist = DemandDistribution(sim_seed)
                
                # Run the simulation and calculate total profit
                df = self.run_single_simulation(q)
                total_profit = df['profit'].sum()
                total_profits.append(total_profit)
            
            results[q] = total_profits
        
        return results

def plot_profit_distribution(monte_carlo_results):
    """Plot distribution of profits for different order quantities."""
    fig = go.Figure()
    
    for q, profits in monte_carlo_results.items():
        fig.add_trace(go.Violin(
            x=[f"Q={q}"] * len(profits),
            y=profits,
            name=f"Q={q}",
            box_visible=True,
            meanline_visible=True
        ))
    
    fig.update_layout(
        title="Profit Distribution by Order Quantity",
        xaxis_title="Order Quantity",
        yaxis_title="Total Profit (pence)",
        template="plotly_white"
    )
    
    return fig

def plot_expected_profit_curve(monte_carlo_results):
    """Plot the expected profit curve based on Monte Carlo results."""
    order_quantities = list(monte_carlo_results.keys())
    mean_profits = [np.mean(profits) for profits in monte_carlo_results.values()]
    std_profits = [np.std(profits) for profits in monte_carlo_results.values()]
    
    # Find the optimal order quantity
    optimal_q = order_quantities[np.argmax(mean_profits)]
    max_profit = max(mean_profits)
    
    fig = go.Figure()
    
    # Add expected profit line
    fig.add_trace(go.Scatter(
        x=order_quantities, 
        y=mean_profits,
        mode='lines+markers',
        name='Expected Profit',
        line=dict(color='royalblue', width=3)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=order_quantities + order_quantities[::-1],
        y=[m + s for m, s in zip(mean_profits, std_profits)] + 
           [m - s for m, s in zip(mean_profits[::-1], std_profits[::-1])],
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev'
    ))
    
    # Highlight optimal point
    fig.add_trace(go.Scatter(
        x=[optimal_q],
        y=[max_profit],
        mode='markers',
        marker=dict(size=12, color='green', symbol='star'),
        name=f'Optimal Q={optimal_q}'
    ))
    
    fig.update_layout(
        title="Expected Profit by Order Quantity",
        xaxis_title="Order Quantity",
        yaxis_title="Expected Total Profit (pence)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def plot_simulation_paths(simulation_dataframes, max_paths=20):
    """Plot individual simulation paths for a given order quantity."""
    # Randomly select a subset of simulations to display
    if len(simulation_dataframes) > max_paths:
        indices = np.random.choice(len(simulation_dataframes), max_paths, replace=False)
        paths_to_show = [simulation_dataframes[i] for i in indices]
    else:
        paths_to_show = simulation_dataframes
    
    fig = go.Figure()
    
    # Add individual simulation paths
    for i, df in enumerate(paths_to_show):
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df['profit'].cumsum(),
            mode='lines',
            opacity=0.5,
            line=dict(width=1),
            name=f'Simulation {i+1}'
        ))
    
    # Add mean path
    all_days = set()
    for df in simulation_dataframes:
        all_days.update(df['day'])
    days = sorted(all_days)
    
    mean_profits = []
    for day in days:
        day_profits = [df[df['day'] == day]['profit'].sum() for df in simulation_dataframes if day in df['day'].values]
        mean_profits.append(np.mean(day_profits))
    
    cumulative_mean = np.cumsum(mean_profits)
    
    fig.add_trace(go.Scatter(
        x=days,
        y=cumulative_mean,
        mode='lines',
        line=dict(color='red', width=3),
        name='Mean Path'
    ))
    
    fig.update_layout(
        title="Cumulative Profit Paths",
        xaxis_title="Day",
        yaxis_title="Cumulative Profit (pence)",
        template="plotly_white"
    )
    
    return fig

def add_sensitivity_analysis_tab():
    return dbc.Tab([
        dbc.Row([
            dbc.Col([
                html.H4("Sensitivity Analysis", className="mt-3"),
                html.P("See how optimal order quantity changes with different parameters"),
                dbc.Select(
                    id="sensitivity-param",
                    options=[
                        {"label": "Purchase Price", "value": "purchase_price"},
                        {"label": "Selling Price", "value": "selling_price"},
                        {"label": "Salvage Value", "value": "salvage_value"}
                    ],
                    value="purchase_price"
                ),
                dbc.Button("Run Analysis", id="sensitivity-button", color="success", className="mt-2")
            ], width=12)
        ]),
        dbc.Spinner([
            dcc.Graph(id="sensitivity-graph")
        ])
    ], label="Sensitivity Analysis")

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Newsvendor Monte Carlo Simulation", className="text-center my-4"),
            html.P("Optimize inventory decisions using Monte Carlo simulation", className="text-center lead"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Parameters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Purchase Price (pence)"),
                            dcc.Input(id="purchase-price", type="number", value=30, min=1, className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("Selling Price (pence)"),
                            dcc.Input(id="selling-price", type="number", value=75, min=1, className="form-control")
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Salvage Value (pence)"),
                            dcc.Input(id="salvage-value", type="number", value=5, min=0, className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("Batch Size"),
                            dcc.Input(id="batch-size", type="number", value=10, min=1, className="form-control")
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Planning Period (days)"),
                            dcc.Input(id="planning-days", type="number", value=20, min=1, className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("Number of Simulations"),
                            dcc.Input(id="num-simulations", type="number", value=500, min=10, max=2000, className="form-control")
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Min Order Quantity"),
                            dcc.Input(id="min-order", type="number", value=10, min=0, className="form-control")
                        ], width=4),
                        dbc.Col([
                            html.Label("Max Order Quantity"),
                            dcc.Input(id="max-order", type="number", value=100, min=10, className="form-control")
                        ], width=4),
                        dbc.Col([
                            html.Label("Step Size"),
                            dcc.Input(id="step-size", type="number", value=10, min=1, className="form-control")
                        ], width=4)
                    ], className="mb-3"),
                    
                    dbc.Button("Run Simulation", id="run-button", color="primary", className="mt-3 w-100")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Spinner([
                dcc.Graph(id="profit-curve-graph")
            ])
        ], width=12, className="mb-4")
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Spinner([
                dcc.Graph(id="profit-distribution-graph")
            ])
        ], width=6),
        dbc.Col([
            dbc.Spinner([
                dcc.Graph(id="simulation-paths-graph")
            ])
        ], width=6)
    ]),
    
    html.Div(id="simulation-results-store", style={"display": "none"})
], fluid=True)

# Define callbacks
@app.callback(
    [Output("profit-curve-graph", "figure"),
     Output("profit-distribution-graph", "figure"),
     Output("simulation-paths-graph", "figure"),
     Output("simulation-results-store", "children")],
    [Input("run-button", "n_clicks")],
    [State("purchase-price", "value"),
     State("selling-price", "value"),
     State("salvage-value", "value"),
     State("batch-size", "value"),
     State("planning-days", "value"),
     State("num-simulations", "value"),
     State("min-order", "value"),
     State("max-order", "value"),
     State("step-size", "value")]
)
def run_simulation(n_clicks, purchase_price, selling_price, salvage_value, 
                  batch_size, planning_days, num_simulations, min_order, max_order, step_size):
    if n_clicks is None:
        # Return empty figures on initial load
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "Run the simulation to see results",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
        return empty_fig, empty_fig, empty_fig, ""
    
    # Create parameters
    params = NewsvendorParams(
        purchase_price=purchase_price,
        selling_price=selling_price,
        salvage_value=salvage_value,
        batch_size=batch_size,
        planning_days=planning_days
    )
    
    # Create simulation
    simulation = NewsvendorSimulation(params=params, seed=42)
    
    # Define order quantities to test
    order_quantities = list(range(min_order, max_order + 1, step_size))
    order_quantities = [q for q in order_quantities if q % batch_size == 0]
    
    # Run Monte Carlo simulation
    monte_carlo_results = simulation.run_monte_carlo(order_quantities, num_simulations)
    
    # Find optimal order quantity
    mean_profits = {q: np.mean(profits) for q, profits in monte_carlo_results.items()}
    optimal_q = max(mean_profits, key=mean_profits.get)
    
    # Generate detailed simulation data for the optimal quantity
    simulation_dataframes = []
    for i in range(min(20, num_simulations)):  # Limit to 20 simulations for the paths plot
        simulation.demand_dist = DemandDistribution(seed=42 + i)
        df = simulation.run_single_simulation(optimal_q)
        simulation_dataframes.append(df)
    
    # Create plots
    profit_curve_fig = plot_expected_profit_curve(monte_carlo_results)
    profit_dist_fig = plot_profit_distribution(monte_carlo_results)
    paths_fig = plot_simulation_paths(simulation_dataframes)
    
    return profit_curve_fig, profit_dist_fig, paths_fig, "Simulation completed"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)



# Streamlit App Configuration
st.set_page_config(page_title="Newsvendor Simulation", layout="wide")

# Sidebar Controls
with st.sidebar:
    st.header("Simulation Parameters")
    purchase_price = st.number_input("Purchase Price (pence)", value=30, min=1)
    selling_price = st.number_input("Selling Price (pence)", value=75, min=1)
    salvage_value = st.number_input("Salvage Value (pence)", value=5, min=0)
    batch_size = st.number_input("Batch Size", value=10, min=1)
    planning_days = st.number_input("Planning Period (days)", value=20, min=1)
    num_sims = st.slider("Number of Simulations", 100, 5000, 500)
    min_order = st.number_input("Min Order Quantity", value=10, min=0)
    max_order = st.number_input("Max Order Quantity", value=100, min=10)
    step_size = st.number_input("Step Size", value=10, min=1)

# Run Simulation
if st.button("Run Monte Carlo Simulation"):
    params = NewsvendorParams(
        purchase_price=purchase_price,
        selling_price=selling_price,
        salvage_value=salvage_value,
        batch_size=batch_size,
        planning_days=planning_days
    )
    
    simulation = NewsvendorSimulation(params=params)
    order_quantities = [q for q in range(min_order, max_order+1, step_size) 
                       if q % batch_size == 0]
    
    with st.spinner("Running simulations..."):
        results = simulation.run_monte_carlo(order_quantities, num_sims)
    
    # Display Results
    st.header("Simulation Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_expected_profit_curve(results), use_container_width=True)
    with col2:
        st.plotly_chart(plot_profit_distribution(results), use_container_width=True)
    
    st.plotly_chart(plot_simulation_paths(
        [simulation.run_single_simulation(max(results, key=lambda k: np.mean(results[k]))) 
         for _ in range(20)]
    ), use_container_width=True)
