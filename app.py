import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

# Set page configuration
st.set_page_config(
    page_title="PINN Heat Transfer Simulator",
    page_icon="🔥",
    layout="wide"
)

# Title and description
st.title("Physics-Informed Neural Network for Heat Transfer Simulation")
st.markdown("""
This application demonstrates a Physics-Informed Neural Network (PINN) that solves the heat equation 
for a pipe with specified boundary conditions. The model was trained to satisfy the heat equation:

$$\\frac{\\partial T}{\\partial t} = \\alpha \\left(\\frac{\\partial^2 T}{\\partial x^2} + \\frac{\\partial^2 T}{\\partial y^2}\\right)$$

With the following boundary conditions:
- Inlet: $T(0,y,t) = T_{inlet}$
- Outlet: $\\frac{\\partial T}{\\partial x}(L,y,t) = 0$
- Inner wall: $T(x,R_{in},t) = T_{fluid}$
- Outer wall: $-k\\frac{\\partial T}{\\partial y}(x,R_{out},t) = h(T - T_{\\infty})$
- Initial condition: $T(x,y,0) = T_{\\infty}$
""")

# Load the PINN model
@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load('pinn_model_complete.pth', map_location=device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, device = load_model()

if model is None:
    st.error("Failed to load the model. Please check if the model file exists.")
    st.stop()

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

# Physical parameters
st.sidebar.subheader("Physical Parameters")
alpha = st.sidebar.number_input("Thermal Diffusivity (α) [m²/s]", value=1.455e-7, format="%.7f")
k = st.sidebar.number_input("Thermal Conductivity (k) [W/(m·K)]", value=45.0)
h = st.sidebar.number_input("Heat Transfer Coefficient (h) [W/(m²·K)]", value=20.0)
T_inlet = st.sidebar.number_input("Inlet Temperature (T_inlet) [K]", value=353.15)
T_inf = st.sidebar.number_input("Ambient Temperature (T_∞) [K]", value=293.15)

# Geometric parameters
st.sidebar.subheader("Geometric Parameters")
L = st.sidebar.slider("Pipe Length (L) [m]", min_value=1.0, max_value=50.0, value=30.0)
R_in = st.sidebar.slider("Inner Radius (R_in) [m]", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
R_out = st.sidebar.slider("Outer Radius (R_out) [m]", min_value=0.02, max_value=0.15, value=0.06, step=0.01)

# Time parameter
st.sidebar.subheader("Time Parameter")
t_final = st.sidebar.slider("Simulation Time (t) [s]", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# Visualization parameters
st.sidebar.subheader("Visualization Parameters")
N_eval = st.sidebar.slider("Resolution", min_value=50, max_value=300, value=200)
colormap = st.sidebar.selectbox("Colormap", options=["viridis", "plasma", "inferno", "magma", "jet", "rainbow"], index=0)

# Function to evaluate the model
def evaluate_model(model, L, R_out, t_final, N_eval, device):
    # Create evaluation points for x and y directions
    x_eval = torch.linspace(0, L, N_eval).unsqueeze(1)
    y_eval = torch.linspace(0, R_out, N_eval).unsqueeze(1)
    
    # Create a single time tensor for t_final
    t_eval = torch.tensor([[t_final]])
    
    # Create a meshgrid over (x, y)
    xx, yy = torch.meshgrid(x_eval.squeeze(), y_eval.squeeze(), indexing='ij')
    xx = xx.reshape(-1, 1)  # Flatten to a column vector
    yy = yy.reshape(-1, 1)
    
    # Repeat the time tensor to match the number of spatial points
    t_grid = t_eval.repeat(xx.shape[0], 1)
    
    # Concatenate the spatial and time inputs
    inp_eval = torch.cat([xx.to(device), yy.to(device), t_grid.to(device)], dim=1)
    
    # Evaluate the model
    with torch.no_grad():
        T_eval = model(inp_eval)
    
    # Move predictions back to CPU and convert to numpy
    T_eval = T_eval.detach().cpu().numpy()
    
    # Reshape T_eval to a grid form for visualization
    T_eval_grid = T_eval.reshape(N_eval, N_eval)
    
    return T_eval_grid, xx.cpu().numpy(), yy.cpu().numpy()

# Main content area
st.header("Temperature Distribution Visualization")

# Evaluate the model with current parameters
T_eval_grid, xx, yy = evaluate_model(model, L, R_out, t_final, N_eval, device)

# Create two columns for the plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("2D Contour Plot")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    contour = ax1.contourf(
        np.linspace(0, L, N_eval),
        np.linspace(0, R_out, N_eval),
        T_eval_grid,
        levels=50,
        cmap=colormap
    )
    plt.colorbar(contour, ax=ax1, label="Temperature [K]")
    ax1.set_title(f"Temperature Distribution at t = {t_final} s")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    
    # Add a horizontal line at y = R_in to show the inner wall
    ax1.axhline(y=R_in, color='r', linestyle='--', label=f"Inner Wall (R_in = {R_in} m)")
    ax1.legend()
    
    st.pyplot(fig1)

with col2:
    st.subheader("3D Surface Plot")
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(np.linspace(0, L, N_eval), np.linspace(0, R_out, N_eval))
    surf = ax2.plot_surface(
        X, Y, T_eval_grid.T,
        cmap=colormap,
        linewidth=0,
        antialiased=True,
        norm=Normalize(vmin=T_eval_grid.min(), vmax=T_eval_grid.max())
    )
    
    ax2.set_title(f"3D Temperature Distribution at t = {t_final} s")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("Temperature [K]")
    
    # Add a colorbar
    fig2.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label="Temperature [K]")
    
    st.pyplot(fig2)

# Temperature profiles
st.header("Temperature Profiles")

# Create two columns for the profile plots
col3, col4 = st.columns(2)

with col3:
    st.subheader("Temperature Profile along x-axis")
    # Select y-position for the profile
    y_pos = st.slider("y-position [m]", min_value=0.0, max_value=float(R_out), value=float(R_in), step=0.01)
    
    # Find the closest y-index
    y_idx = int(y_pos / R_out * (N_eval - 1))
    
    # Plot temperature profile along x at selected y
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(np.linspace(0, L, N_eval), T_eval_grid[:, y_idx], 'b-', linewidth=2)
    ax3.set_title(f"Temperature Profile along x at y = {y_pos:.3f} m, t = {t_final} s")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("Temperature [K]")
    ax3.grid(True)
    
    st.pyplot(fig3)

with col4:
    st.subheader("Temperature Profile along y-axis")
    # Select x-position for the profile
    x_pos = st.slider("x-position [m]", min_value=0.0, max_value=float(L), value=float(L/2), step=1.0)
    
    # Find the closest x-index
    x_idx = int(x_pos / L * (N_eval - 1))
    
    # Plot temperature profile along y at selected x
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(np.linspace(0, R_out, N_eval), T_eval_grid[x_idx, :], 'r-', linewidth=2)
    ax4.axvline(x=R_in, color='k', linestyle='--', label=f"Inner Wall (R_in = {R_in} m)")
    ax4.set_title(f"Temperature Profile along y at x = {x_pos:.3f} m, t = {t_final} s")
    ax4.set_xlabel("y [m]")
    ax4.set_ylabel("Temperature [K]")
    ax4.grid(True)
    ax4.legend()
    
    st.pyplot(fig4)

# Add information about the model
st.header("About the PINN Model")
st.markdown("""
### Physics-Informed Neural Network (PINN)
This model uses a neural network to solve the heat equation with specific boundary conditions for a pipe. 
The network is trained to minimize both the physics residual (the PDE) and the boundary condition errors.

### Model Architecture
- Input: 3 features (x, y, t)
- Hidden layers: 3 layers with 32 neurons each
- Activation function: Tanh
- Output: Temperature T(x, y, t)

### Training Process
The model was trained by minimizing a weighted sum of losses:
- PDE residual loss
- Inlet boundary condition loss
- Outlet boundary condition loss
- Inner wall boundary condition loss
- Outer wall boundary condition loss
- Initial condition loss
""")

# Footer
st.markdown("---")
st.markdown("PINN Heat Transfer Simulator - Created with Streamlit")