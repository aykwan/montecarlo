import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# Streamlit UI
st.title("Newsvendor Monte Carlo Simulation")

with st.sidebar:
    st.header("Simulation Parameters")
    purchase_price = st.number_input("Purchase Price (pence)", value=30)
    selling_price = st.number_input("Selling Price (pence)", value=75)
    salvage_value = st.number_input("Salvage Value (pence)", value=5)
    batch_size = st.number_input("Batch Size", value=10)
    planning_days = st.number_input("Planning Period (days)", value=20)
    num_simulations = st.number_input("Number of Simulations", value=500)
    min_order = st.number_input("Min Order Quantity", value=10)
    max_order = st.number_input("Max Order Quantity", value=100)
    step_size = st.number_input("Step Size", value=10)

if st.button("Run Simulation"):
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
        results = simulation.run_monte_carlo(order_quantities, num_simulations)
    
    st.plotly_chart(plot_expected_profit_curve(results))
    st.plotly_chart(plot_profit_distribution(results))