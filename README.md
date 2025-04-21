Newsvendor Monte Carlo Simulation
=================================

A Monte Carlo simulation is a computational technique that uses random sampling to estimate the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables.

This project implements a Monte Carlo simulation to solve the classic newsvendor problem, a fundamental challenge in inventory management. The application helps users determine the optimal order quantity for perishable goods under uncertain demand to maximize expected profit.

An online application is available for use following this link:
https://montecarlo-mhogdtncyfruvj44xyf2sk.streamlit.app/

Overview
--------
The newsvendor problem models inventory decisions for perishable products where demand is uncertain. This project uses Monte Carlo simulation to:
1. Generate random demand scenarios based on historical data or probability distributions.
2. Simulate profits for different order quantities.
3. Identify the order quantity that maximizes expected profit.
**4. A web-based Streamlit is available for viewing and manipulating. Otherwise, the one for release (on the right sidebar) is using Dash instead of Streamlit. The guide here shows how to run the Dash application locally on your computer.**__

Prerequisites
-------------
To run this project locally, you need:
- Python 3.8 or higher installed on your system.
- A terminal or command prompt to execute Python scripts.

Installation Instructions
-------------------------
1. Download the project files:
   - Ensure you have all Python scripts and dependencies required for the simulation.

2. Install the required Python libraries:
- numpy
- pandas
- plotly
- dash
- dash-bootstrap-components

3. Run the ipynb file with your choice of IDE (Jupyter, VSCode, etc.)

4. Use the interactive dashboard to:
- Adjust simulation parameters (e.g., purchase price, selling price, salvage value).
- Set the range of order quantities to test.
- Run simulations and view results in real-time.

Simulation Parameters
---------------------
You can customize the following parameters in the dashboard:
- **Purchase Price**: Cost of purchasing one unit of inventory.
- **Selling Price**: Revenue earned from selling one unit of inventory.
- **Salvage Value**: Value recovered from unsold inventory.
- **Batch Size**: Minimum increment in which inventory can be ordered.
- **Planning Period**: Number of days over which simulations are run.
- **Demand Distribution**: Probability distribution of daily demand (e.g., good/fair/poor days).

Visualizations
--------------
The application provides three key visualizations:

1. **Expected Profit Curve**:
- Displays expected profit for different order quantities.
- Highlights the optimal order quantity that maximizes profit.

2. **Profit Distribution**:
- Shows the distribution of profits for each tested order quantity using violin plots.

3. **Simulation Paths**:
- Visualizes cumulative profits over time for multiple simulations at the optimal order quantity.

