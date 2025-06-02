import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

st.set_page_config(
    page_title="Dvorkin Thesis",
    page_icon="🌡️",
    layout="wide"
)

st.title("Integration of Physics-Informed Neural Networks with Thermodynamic Principles: Application in Heat Transfer Systems")
st.markdown("""
Student: Ivan Dvorkin\n
Supervisor: PhD, Associate Professor, Tarakanov Aleksandr Aleksandrovich
""")

@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = [3, 32, 32, 32, 1]
        model = PINN(layers)
        model.load_state_dict(torch.load('pinn_model_complete.pth', map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, device = load_model()

if model is None:
    st.error("Failed to load the model. Please check if the model file exists.")
    st.stop()

st.sidebar.header("Simulation Parameters")

st.sidebar.subheader("Physical Parameters")
alpha = st.sidebar.number_input("Thermal Diffusivity (α) [m²/s]", value=1.455e-7, format="%.7f")
k = st.sidebar.number_input("Thermal Conductivity (k) [W/(m·K)]", value=45.0)
h = st.sidebar.number_input("Heat Transfer Coefficient (h) [W/(m²·K)]", value=20.0)
T_inlet = st.sidebar.number_input("Inlet Temperature (T_inlet) [K]", value=353.15)
T_inf = st.sidebar.number_input("Ambient Temperature (T_∞) [K]", value=293.15)

st.sidebar.subheader("Geometric Parameters")
L = st.sidebar.slider("Pipe Length (L) [m]", min_value=1.0, max_value=50.0, value=30.0)
R_in = st.sidebar.slider("Inner Radius (R_in) [m]", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
R_out = st.sidebar.slider("Outer Radius (R_out) [m]", min_value=0.02, max_value=0.15, value=0.06, step=0.01)

st.sidebar.subheader("Time Parameter")
t_final = st.sidebar.slider("Simulation Time (t) [s]", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
update_button = st.sidebar.button("Update Prediction", use_container_width=True)

N_eval = 200
colormap = "viridis"


def evaluate_model(model, L, R_out, t_final, N_eval, device):
    x_eval = torch.linspace(0, L, N_eval).unsqueeze(1)
    y_eval = torch.linspace(0, R_out, N_eval).unsqueeze(1)

    t_eval = torch.tensor([[t_final]])

    xx, yy = torch.meshgrid(x_eval.squeeze(), y_eval.squeeze(), indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)

    t_grid = t_eval.repeat(xx.shape[0], 1)

    inp_eval = torch.cat([xx.to(device), yy.to(device), t_grid.to(device)], dim=1)

    with torch.no_grad():
        T_eval = model(inp_eval)

    T_eval = T_eval.detach().cpu().numpy()

    T_eval_grid = T_eval.reshape(N_eval, N_eval)

    return T_eval_grid, xx.cpu().numpy(), yy.cpu().numpy()


if 'T_eval_grid' not in st.session_state:
    st.session_state.T_eval_grid, st.session_state.xx, st.session_state.yy = evaluate_model(model, L, R_out, t_final, N_eval, device)

if update_button:
    st.session_state.T_eval_grid, st.session_state.xx, st.session_state.yy = evaluate_model(model, L, R_out, t_final, N_eval, device)


st.header("Temperature Distribution Gradient")

T_eval_grid = st.session_state.T_eval_grid
xx = st.session_state.xx
yy = st.session_state.yy


fig1, ax1 = plt.subplots(figsize=(5, 3))
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

ax1.axhline(y=R_in, color='r', linestyle='--', label=f"Inner Wall (R_in = {R_in} m)")
ax1.legend()

st.pyplot(fig1)

st.header("Temperature Profiles")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Temperature Profile along x-axis")
    y_pos = st.slider("y-position [m]", min_value=0.0, max_value=float(R_out), value=float(R_in), step=0.01)

    y_idx = int(y_pos / R_out * (N_eval - 1))
    y_idx = max(0, min(y_idx, N_eval - 1))

    print(y_pos)
    print(R_out)
    print(N_eval)
    print(y_idx)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(np.linspace(0, L, N_eval), T_eval_grid[:, y_idx], 'b-', linewidth=2)
    ax3.set_title(f"Temperature Profile along x at y = {y_pos:.3f} m, t = {t_final} s")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("Temperature [K]")
    ax3.grid(True)

    st.pyplot(fig3)

with col4:
    st.subheader("Temperature Profile along y-axis")
    x_pos = st.slider("x-position [m]", min_value=0.0, max_value=float(L), value=float(L/2), step=1.0)

    x_idx = int(x_pos / L * (N_eval - 1))
    x_idx = max(0, min(x_idx, N_eval - 1))

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(np.linspace(0, R_out, N_eval), T_eval_grid[x_idx, :], 'r-', linewidth=2)
    ax4.axvline(x=R_in, color='k', linestyle='--', label=f"Inner Wall (R_in = {R_in} m)")
    ax4.set_title(f"Temperature Profile along y at x = {x_pos:.3f} m, t = {t_final} s")
    ax4.set_xlabel("y [m]")
    ax4.set_ylabel("Temperature [K]")
    ax4.grid(True)
    ax4.legend()

    st.pyplot(fig4)

st.markdown("---")
st.markdown("HSE 2025")
